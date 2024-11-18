import torch
from pytorch3d.renderer import (
    FoVPerspectiveCameras, 
    NDCMultinomialRaysampler,
    MonteCarloRaysampler,
    EmissionAbsorptionRaymarcher,
    ImplicitRenderer,
    RayBundle,
    ray_bundle_to_ray_points,
)
class HarmonicEmbedding(torch.nn.Module):
    def __init__(self, n_harmonic_functions=60, omega0=0.1):
        """
        Given an input tensor `x` of shape [minibatch, ... , dim],
        the harmonic embedding layer converts each feature
        in `x` into a series of harmonic features `embedding`
        as follows:
            embedding[..., i*dim:(i+1)*dim] = [
                sin(x[..., i]),
                sin(2*x[..., i]),
                sin(4*x[..., i]),
                ...
                sin(2**(self.n_harmonic_functions-1) * x[..., i]),
                cos(x[..., i]),
                cos(2*x[..., i]),
                cos(4*x[..., i]),
                ...
                cos(2**(self.n_harmonic_functions-1) * x[..., i])
            ]
            
        Note that `x` is also premultiplied by `omega0` before
        evaluating the harmonic functions.
        """
        super().__init__()
        self.register_buffer(
            'frequencies',
            omega0 * (2.0 ** torch.arange(n_harmonic_functions)),
        )
    def forward(self, x):
        """
        Args:
            x: tensor of shape [..., dim]
        Returns:
            embedding: a harmonic embedding of `x`
                of shape [..., n_harmonic_functions * dim * 2]
        """
        embed = (x[..., None] * self.frequencies).view(*x.shape[:-1], -1)
        return torch.cat((embed.sin(), embed.cos()), dim=-1)

class OccupancyGrid:
    def __init__(self, grid_resolution=32, threshold=0.5, volume_extent=3.0, device="cpu"):
        """
        Initializes the occupancy grid with a given resolution, threshold,
        and extent of the volume in world coordinates.
        """
        self.grid_resolution = grid_resolution
        self.threshold = threshold
        self.volume_extent = volume_extent
        self.voxel_size = 2 * volume_extent / grid_resolution
        self.device = device
        # Initialize the occupancy grid as a boolean tensor on the specified device
        self.grid = torch.zeros((grid_resolution,) * 3, dtype=torch.bool, device=self.device)
        
    def update(self, density_samples, points_world):
        """
        Updates the occupancy grid by marking voxels as occupied if the density
        exceeds a given threshold.
        
        Args:
            density_samples: Tensor of densities for sampled points (should be on the same device).
            points_world: Corresponding world coordinates for density samples (should be on the same device).
        """
        # Ensure points are on the same device
        points_world = points_world.to(self.device)
        voxel_indices = ((points_world + self.volume_extent) / self.voxel_size).long()
        
        # Mask for points with density above the threshold
        mask = (density_samples > self.threshold).squeeze(-1)
        
        # Update grid only at voxel indices where density > threshold
        self.grid[voxel_indices[mask].unbind(dim=-1)] = True
        
    def is_occupied(self, points_world):
        """
        Checks if the given world coordinates lie within occupied voxels.
        
        Args:
            points_world: Tensor of shape [..., 3] representing 3D world points.
        Returns:
            mask: A boolean mask indicating whether each point is in an occupied voxel.
        """
        # Move points to the same device as the grid
        points_world = points_world.to(self.device)
        voxel_indices = ((points_world + self.volume_extent) / self.voxel_size).long()
        
        # Create a mask for occupancy based on voxel indices
        mask = torch.zeros(points_world.shape[:-1], dtype=torch.bool, device=self.device)
        mask = self.grid[voxel_indices.unbind(dim=-1)]
        return mask



class NeuralRadianceField(torch.nn.Module):
    def __init__(self, n_harmonic_functions=30, n_hidden_neurons=64, grid_resolution=32, device="cpu"):
        super().__init__()
        # Initialize occupancy grid with the specified device
        self.device = device
        self.occupancy_grid = OccupancyGrid(grid_resolution=grid_resolution, device=self.device)
        
        # Existing harmonic embedding and MLP layers
        self.harmonic_embedding = HarmonicEmbedding(n_harmonic_functions).to(self.device)
        embedding_dim = n_harmonic_functions * 2 * 3
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, n_hidden_neurons),
            torch.nn.Softplus(beta=10.0),
            torch.nn.Linear(n_hidden_neurons, n_hidden_neurons),
            torch.nn.Softplus(beta=10.0),
        ).to(self.device)
        
        # Color and density layers
        self.color_layer = torch.nn.Sequential(
            torch.nn.Linear(n_hidden_neurons + embedding_dim, n_hidden_neurons),
            torch.nn.Softplus(beta=10.0),
            torch.nn.Linear(n_hidden_neurons, 3),
            torch.nn.Sigmoid(),
        ).to(self.device)
        self.density_layer = torch.nn.Sequential(
            torch.nn.Linear(n_hidden_neurons, 1),
            torch.nn.Softplus(beta=10.0),
        ).to(self.device)
        self.density_layer[0].bias.data[0] = -1.5

    def _get_densities(self, features):
        raw_densities = self.density_layer(features)
        return 1 - (-raw_densities).exp()

    def _get_colors(self, features, rays_directions):
        rays_directions_normed = torch.nn.functional.normalize(rays_directions, dim=-1).to(self.device)
        rays_embedding = self.harmonic_embedding(rays_directions_normed)
        rays_embedding_expand = rays_embedding[..., None, :].expand(
            *features.shape[:-1], rays_embedding.shape[-1]
        )
        color_layer_input = torch.cat((features, rays_embedding_expand), dim=-1)
        return self.color_layer(color_layer_input)
    
    def forward(self, ray_bundle: RayBundle, **kwargs):
        rays_points_world = ray_bundle_to_ray_points(ray_bundle).to(self.device)
        occupancy_mask = self.occupancy_grid.is_occupied(rays_points_world)
        
        if not occupancy_mask.any():
            # Return zeros if all regions are empty
            return (torch.zeros(*rays_points_world.shape[:-1], 1, device=self.device, requires_grad=True),
                    torch.zeros(*rays_points_world.shape[:-1], 3, device=self.device, requires_grad=True))
        
        # Process only occupied points
        embeds = self.harmonic_embedding(rays_points_world[occupancy_mask])
        features = self.mlp(embeds)
        rays_densities = torch.zeros_like(occupancy_mask, dtype=torch.float32, device=self.device, requires_grad=True)
        rays_colors = torch.zeros_like(occupancy_mask, dtype=torch.float32, device=self.device, requires_grad=True)
        
        rays_densities[occupancy_mask] = self._get_densities(features)
        rays_colors[occupancy_mask] = self._get_colors(features, ray_bundle.directions[occupancy_mask])
        
        return rays_densities, rays_colors

    def update_occupancy_grid(self, ray_bundle: RayBundle):
        rays_points_world = ray_bundle_to_ray_points(ray_bundle).to(self.device)
        embeds = self.harmonic_embedding(rays_points_world)
        features = self.mlp(embeds)
        densities = self._get_densities(features)
        self.occupancy_grid.update(densities, rays_points_world)

    def batched_forward(
        self, 
        ray_bundle: RayBundle,
        n_batches: int = 16,
        **kwargs,        
    ):
        """
        A batched version of forward() to process large batches in smaller chunks.
        """
        n_pts_per_ray = ray_bundle.lengths.shape[-1]  
        spatial_size = [*ray_bundle.origins.shape[:-1], n_pts_per_ray]
        tot_samples = ray_bundle.origins.shape[:-1].numel()
        batches = torch.chunk(torch.arange(tot_samples), n_batches)

        batch_outputs = [
            self.forward(
                RayBundle(
                    origins=ray_bundle.origins.view(-1, 3)[batch_idx].to(self.device),
                    directions=ray_bundle.directions.view(-1, 3)[batch_idx].to(self.device),
                    lengths=ray_bundle.lengths.view(-1, n_pts_per_ray)[batch_idx].to(self.device),
                    xys=None,
                )
            ) for batch_idx in batches
        ]
        
        rays_densities, rays_colors = [
            torch.cat(
                [batch_output[output_i] for batch_output in batch_outputs], dim=0
            ).view(*spatial_size, -1) for output_i in (0, 1)
        ]
        return rays_densities, rays_colors
