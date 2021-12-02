import torch
from .volumetric_rendering import *
from .generators import ImplicitGenerator3d


def calc_depth_var(rgb_sigma, z_vals, device, noise_std=0.5, last_back=False, white_back=False, clamp_mode=None, pred_occ=False, pred_depth=None):
    sigmas = rgb_sigma[..., 3:]

    if pred_occ: # use occupancy representation
        alphas = torch.sigmoid(sigmas)
        if pred_depth is not None:
            new_alphas = torch.zeros_like(alphas)
            mask = z_vals > pred_depth
            new_alphas[mask] = alphas[mask]
            alphas = new_alphas
    else:
        deltas = z_vals[:, :, 1:] - z_vals[:, :, :-1]
        delta_inf = 1e10 * torch.ones_like(deltas[:, :, :1])
        deltas = torch.cat([deltas, delta_inf], -2)

        noise = torch.randn(sigmas.shape, device=device) * noise_std

        if clamp_mode == 'softplus':
            alphas = 1 - torch.exp(-deltas * (F.softplus(sigmas + noise)))
        elif clamp_mode == 'relu':
            alphas = 1 - torch.exp(-deltas * (F.relu(sigmas + noise)))
        else:
            raise "Need to choose clamp mode"

    alphas_shifted = torch.cat([torch.ones_like(alphas[:, :, :1]), 1-alphas + 1e-10], -2)
    weights = alphas * torch.cumprod(alphas_shifted, -2)[:, :, :-1]
    weights_sum = weights.sum(2)

    if last_back:
        weights[:, :, -1] += (1 - weights_sum)

    depth_final = torch.sum(weights * z_vals, -2) / (weights_sum + 1e-5)
    n = z_vals.size(2)
    assert weights.size(0) == 1
    widx = torch.nonzero((weights_sum > 0.99).squeeze()).squeeze()
    depth_var = torch.sum(weights.index_select(1, widx) * (z_vals.index_select(1, widx) - depth_final.index_select(1, widx).unsqueeze(2))**2, dim=2) / ((weights_sum.index_select(1, widx)+1e-5)*(n-1)/n)
    
    return depth_var

class Generator_Weight(ImplicitGenerator3d):
    def __init__(self, siren, z_dim, **kwargs):
        super().__init__(siren, z_dim, **kwargs)
    
    def get_depth_var(self, z, img_size, fov, ray_start, ray_end, h_stddev, v_stddev, h_mean, v_mean, psi=1, lock_view_dependence=False, max_batch_size=50000, depth_map=False, sample_dist=None, hierarchical_sample=False, **kwargs):
        batch_size = z.shape[0]
        self.generate_avg_frequencies()

        # short hand
        pred_occ = kwargs.get('pred_occ', False)
        surface_sample = kwargs.get('surface_sample', False)
        num_steps_coarse = kwargs.get('num_steps_coarse', 9)
        num_steps_surface = kwargs.get('num_steps_surface', 9)

        with torch.no_grad():
            raw_frequencies, raw_phase_shifts = self.siren.mapping_network(z)

            truncated_frequencies = self.avg_frequencies + psi * (raw_frequencies - self.avg_frequencies)
            truncated_phase_shifts = self.avg_phase_shifts + psi * (raw_phase_shifts - self.avg_phase_shifts)

            points_cam, z_vals, rays_d_cam = get_initial_rays_trig(
                batch_size, num_steps_coarse, resolution=(img_size, img_size), device=self.device, fov=fov, ray_start=ray_start, ray_end=ray_end) # batch_size, pixels, num_steps_coarse, 1
            transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw = transform_sampled_points(
                points_cam, z_vals, rays_d_cam, h_stddev=h_stddev, v_stddev=v_stddev, h_mean=h_mean, v_mean=v_mean, device=self.device, mode=sample_dist)
            transformed_points = transformed_points.reshape(batch_size, img_size*img_size*num_steps_coarse, 3)

            transformed_ray_directions_coarse = self._expand_ray_directions(transformed_ray_directions, num_steps_coarse, lock_view_dependence)
            # Sequentially evaluate siren with max_batch_size to avoid OOM
            coarse_output = torch.zeros((batch_size, transformed_points.shape[1], 4), device=self.device)
            for b in range(batch_size):
                head = 0
                while head < transformed_points.shape[1]:
                    tail = head + max_batch_size
                    coarse_output[b:b+1, head:tail] = self.siren.forward_with_frequencies_phase_shifts(
                        transformed_points[b:b+1, head:tail], truncated_frequencies[b:b+1], truncated_phase_shifts[b:b+1], ray_directions=transformed_ray_directions_coarse[b:b+1, head:tail])
                    head += max_batch_size

            coarse_output = coarse_output.reshape(batch_size, img_size * img_size, num_steps_coarse, 4)

            if surface_sample:
                # surface_z_vals shape: [batch_size, num_pixels**2, num_steps_surface, 1]
                _, pred_depth, _ = self.perform_ray_marching(
                    coarse_output,
                    z_vals,
                    transformed_ray_origins,
                    transformed_ray_directions,
                    n_steps=num_steps_coarse,
                    n_samples=num_steps_surface,
                    interval=kwargs['interval'],
                    z=z,
                    n_secant_steps=kwargs.get('n_secant_steps', 8),
                    tau=kwargs.get('tau', 0.5),
                    truncated_frequencies=truncated_frequencies,
                    truncated_phase_shifts=truncated_phase_shifts
                )                

            depth_var = calc_depth_var(
                coarse_output,
                z_vals,
                device=self.device,
                white_back=kwargs.get('white_back', False),
                clamp_mode = kwargs['clamp_mode'],
                last_back=kwargs.get('last_back', False),
                noise_std=kwargs['nerf_noise'],
                pred_occ=pred_occ,
                pred_depth=pred_depth.unsqueeze(-1).unsqueeze(-1) - kwargs['interval_max'] / 2 if surface_sample else None,
                # clean useless points in front of the surface since they are not well-trained especially in the begining.
            )

        depth_var = depth_var.squeeze().cpu().numpy().mean()
        torch.cuda.empty_cache() # clear the middle cache to free memory

        return depth_var