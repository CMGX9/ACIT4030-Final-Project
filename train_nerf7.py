import torch
import matplotlib.pyplot as plt

from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    NDCMultinomialRaysampler,
    MonteCarloRaysampler,
    EmissionAbsorptionRaymarcher,
    ImplicitRenderer,
)
from plot_image_grid import image_grid
from generate_cow_renders import generate_cow_renders
from helper_functions import (generate_rotating_nerf,
                                    huber,
                                    show_full_render,
                                    sample_images_at_mc_locs)
from nerf_model7 import NeuralRadianceField

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
    print(f"Warning: Using CPU. Will be slow!")

target_cameras, target_images, target_silhouettes = generate_cow_renders(num_views=40, azimuth_range=180) 
print(f'Generated {len(target_images)} images/silhouettes/cameras.')

render_size = target_images.shape[1] * 2

volume_extent_world = 3.0

raysampler_mc = MonteCarloRaysampler(
    min_x = -1.0,
    max_x = 1.0,
    min_y = -1.0,
    max_y = 1.0,
    n_rays_per_image=750,
    n_pts_per_ray=128,
    min_depth=0.1,
    max_depth=volume_extent_world,
)

raymarcher = EmissionAbsorptionRaymarcher()
renderer_mc = ImplicitRenderer(raysampler=raysampler_mc, raymarcher=raymarcher)

render_size = target_images.shape[1] * 2
volume_extent_world = 3.0
raysampler_grid = NDCMultinomialRaysampler(
    image_height=render_size,
    image_width=render_size,
    n_pts_per_ray=128,
    min_depth=0.1,
    max_depth=volume_extent_world)

raymarcher = EmissionAbsorptionRaymarcher()
renderer_grid = ImplicitRenderer(
    raysampler=raysampler_grid, raymarcher=raymarcher)

neural_radiance_field = NeuralRadianceField(device=device)

torch.manual_seed(1)
renderer_grid = renderer_grid.to(device)
renderer_mc = renderer_mc.to(device)
target_cameras = target_cameras.to(device)
target_images = target_images.to(device)
target_silhouettes = target_silhouettes.to(device)
neural_radiance_field = neural_radiance_field.to(device)

lr = 1e-3
optimizer = torch.optim.Adam(neural_radiance_field.parameters(), lr=lr) 
batch_size = 6
n_iter = 3000

loss_history_color, loss_history_sil = [], []
for iteration in range(n_iter):
    print(f'iteration: {iteration}')
    
    if iteration == round(n_iter * 0.75):
        optimizer = torch.optim.Adam(neural_radiance_field.parameters(), lr=lr * 0.1)

    optimizer.zero_grad()
    batch_idx = torch.randperm(len(target_cameras))[:batch_size]

    batch_cameras = FoVPerspectiveCameras(
        R=target_cameras.R[batch_idx].to(device),
        T=target_cameras.T[batch_idx].to(device),
        znear=target_cameras.znear[batch_idx].to(device),
        zfar=target_cameras.zfar[batch_idx].to(device),
        aspect_ratio=target_cameras.aspect_ratio[batch_idx].to(device),
        fov=target_cameras.fov[batch_idx].to(device),
        device=device
    )
    
    rendered_images_silhouettes, sampled_rays = renderer_mc(
        cameras=batch_cameras,
        volumetric_function=neural_radiance_field
    )
    
    rendered_images, rendered_silhouettes = rendered_images_silhouettes.split([3, 1], dim=-1)
    silhouettes_at_rays = sample_images_at_mc_locs(target_silhouettes[batch_idx, ..., None].to(device), sampled_rays.xys)
    sil_err = huber(rendered_silhouettes, silhouettes_at_rays).abs().mean()
    colors_at_rays = sample_images_at_mc_locs(target_images[batch_idx].to(device), sampled_rays.xys)
    color_err = huber(rendered_images, colors_at_rays).abs().mean()
    loss = color_err + sil_err
    loss_history_color.append(float(color_err))
    loss_history_sil.append(float(sil_err))
     # Check if loss requires gradients (should be True)
    if loss.requires_grad:
        loss.backward()
        optimizer.step()
    else:
        print("Error: Loss does not require gradients, check tensor requires_grad settings.")
    

    if iteration % 50 == 0:
        neural_radiance_field.update_occupancy_grid(sampled_rays)
        
    if iteration % 100 == 0:
        fig = show_full_render(neural_radiance_field, batch_cameras, target_images[batch_idx][0], target_silhouettes[batch_idx][0], renderer_grid, loss_history_color, loss_history_sil)
        fig.savefig(f'intermediate_{iteration}')

with torch.no_grad():
    rotating_nerf_frames = generate_rotating_nerf(neural_radiance_field, target_cameras, renderer_grid, n_frames=3*5, device=device) 

image_grid(rotating_nerf_frames.clamp(0., 1.).cpu().numpy(), rows=3, cols=5, rgb=True, fill=True)
plt.show()
