"""Implicit generator for 3D volumes"""

import torch.nn as nn
import torch
from torch.nn.modules.module import T
import numpy as np

from .volumetric_rendering import *


def get_logits_from_prob(probs, eps=1e-4):
    probs = np.clip(probs, a_min=eps, a_max=1-eps)
    logits = np.log(probs / (1 - probs))
    return logits

class ImplicitGenerator3d(nn.Module):
    def __init__(self, siren, z_dim, **kwargs):
        super().__init__()
        self.z_dim = z_dim
        self.siren = siren(output_dim=4, z_dim=self.z_dim, input_dim=3, device=None)
        self.epoch = 0
        self.step = 0

    def set_device(self, device):
        self.device = device
        self.siren.device = device

        self.generate_avg_frequencies()

    def _expand_ray_directions(self, transformed_ray_directions, num_steps, lock_view_dependence=False):
        # transformed_ray_directions: [batch_size, img_size**2, 3], transformed_ray_origins: same shape
        batch_size = transformed_ray_directions.shape[0]
        transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2).expand(-1, -1, num_steps, -1).reshape(batch_size, -1, 3)
        if lock_view_dependence:
            transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
            transformed_ray_directions_expanded[..., -1] = -1

        return transformed_ray_directions_expanded

    def forward(self, z, img_size, fov, ray_start, ray_end, h_stddev, v_stddev, h_mean, v_mean, hierarchical_sample, sample_dist=None, lock_view_dependence=False, **kwargs):
        """
        Generates images from a noise vector, rendering parameters, and camera distribution.
        Uses the hierarchical sampling scheme described in NeRF.
        """

        batch_size = z.shape[0]
        outputs = {}

        all_outputs = []
        all_z_vals = []

        # short hand
        pred_occ = kwargs.get('pred_occ', False)
        surface_sample = kwargs.get('surface_sample', False)
        with_normal_loss = kwargs.get('with_normal_loss', False)
        with_tvprior = kwargs.get('with_tvprior', False)
        with_opacprior = kwargs.get('with_opacprior', False)
        num_steps_coarse = kwargs.get('num_steps_coarse', 9)
        num_steps_surface = kwargs.get('num_steps_surface', 9)
        num_steps_fine = kwargs.get('num_steps_fine', 6)

        # Generate initial camera rays and sample points.
        with torch.no_grad():
            points_cam, z_vals, rays_d_cam = get_initial_rays_trig(
                batch_size, num_steps_coarse, resolution=(img_size, img_size), device=self.device, fov=fov, ray_start=ray_start, ray_end=ray_end) # batch_size, pixels, num_steps_coarse, 1
            transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw = transform_sampled_points(
                points_cam, z_vals, rays_d_cam, h_stddev=h_stddev, v_stddev=v_stddev, h_mean=h_mean, v_mean=v_mean, device=self.device, mode=sample_dist)
            transformed_points = transformed_points.reshape(batch_size, img_size*img_size*num_steps_coarse, 3)

        # Model prediction on course points
        if not surface_sample:
            # transformed_points, transformed_ray_directions_expanded, coarse_output: torch.float16; z: torch.float32
            coarse_output = self.siren(transformed_points, z, ray_directions=self._expand_ray_directions(transformed_ray_directions, num_steps_coarse, lock_view_dependence))
            coarse_output = coarse_output.reshape(batch_size, img_size * img_size, num_steps_coarse, 4)
            all_outputs = [coarse_output]
            all_z_vals = [z_vals]
        else:
            with torch.no_grad():
                coarse_output = self.siren(transformed_points, z, ray_directions=self._expand_ray_directions(transformed_ray_directions, num_steps_coarse, lock_view_dependence))
                coarse_output = coarse_output.reshape(batch_size, img_size * img_size, num_steps_coarse, 4)

                # surface_z_vals shape: [batch_size, num_pixels**2, num_steps_surface, 1]
                surface_z_vals, pred_depth, mask = self.perform_ray_marching(
                    coarse_output.detach(),
                    z_vals,
                    transformed_ray_origins,
                    transformed_ray_directions,
                    n_steps=num_steps_coarse,
                    n_samples=num_steps_surface,
                    z=z,
                    interval=kwargs['interval'],
                    n_secant_steps=kwargs.get('n_secant_steps', 8),
                    method=kwargs.get('method', 'secant'),
                    tau=kwargs.get('tau', 0.5),
                    tohalf=True
                )
                surface_z_vals = surface_z_vals.detach()
                surface_around_points = transformed_ray_origins.unsqueeze(2).contiguous() + transformed_ray_directions.unsqueeze(2).contiguous() * surface_z_vals.expand(-1,-1,-1,3).contiguous()
                surface_around_points, surface_z_vals = perturb_points(surface_around_points, surface_z_vals, transformed_ray_directions, self.device)
                surface_around_points = surface_around_points.reshape(batch_size, -1, 3)
                outputs['mask'] = mask
                outputs['pred_depth'] = pred_depth.detach().cpu().clamp(min=ray_start, max=ray_end)
            
            surface_output = self.siren(surface_around_points, z, ray_directions=self._expand_ray_directions(transformed_ray_directions, num_steps_surface, lock_view_dependence))
            surface_output = surface_output.reshape(batch_size, img_size * img_size, -1, 4)

            all_outputs.append(surface_output)
            all_z_vals.append(surface_z_vals)
            
        # Re-sample fine points alont camera rays, as described in NeRF
        if hierarchical_sample:
            with torch.no_grad():
                transformed_points = transformed_points.reshape(batch_size, img_size * img_size, num_steps_coarse, 3)
                weights = fancy_integration(
                    coarse_output,
                    z_vals,
                    device=self.device,
                    clamp_mode=kwargs['clamp_mode'],
                    noise_std=kwargs['nerf_noise'],
                    pred_occ=pred_occ
                )['weights']

                weights = weights.reshape(batch_size * img_size * img_size, num_steps_coarse) + 1e-5

                #### Start new importance sampling
                z_vals = z_vals.reshape(batch_size * img_size * img_size, num_steps_coarse)
                z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:])
                z_vals = z_vals.reshape(batch_size, img_size * img_size, num_steps_coarse, 1)
                fine_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1], num_steps_fine, det=False).detach()
                fine_z_vals = fine_z_vals.reshape(batch_size, img_size * img_size, num_steps_fine, 1)

                fine_points = transformed_ray_origins.unsqueeze(2).contiguous() + transformed_ray_directions.unsqueeze(2).contiguous() * fine_z_vals.expand(-1,-1,-1,3).contiguous()
                fine_points = fine_points.reshape(batch_size, img_size*img_size*num_steps_fine, 3)
                    
                #### end new importance sampling

            # Model prediction on re-sampled find points
            fine_output = self.siren(fine_points, z, ray_directions=self._expand_ray_directions(transformed_ray_directions, num_steps_fine, lock_view_dependence))
            fine_output = fine_output.reshape(batch_size, img_size * img_size, -1, 4)

            # Combine course and fine points
            all_outputs.append(fine_output)
            all_z_vals.append(fine_z_vals)

        # combine all the outputs
        all_outputs = torch.cat(all_outputs, dim=-2)
        all_z_vals = torch.cat(all_z_vals, dim=-2)
        _, indices = torch.sort(all_z_vals, dim=-2)
        all_z_vals = torch.gather(all_z_vals, -2, indices)
        # Target sizes: [-1, -1, -1, 4].  Tensor sizes: [240, 512, 12]
        all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, 4))

        # Create images with NeRF
        integ_results = fancy_integration(
            all_outputs,
            all_z_vals,
            device=self.device,
            white_back=kwargs.get('white_back', False),
            last_back=kwargs.get('last_back', False),
            clamp_mode=kwargs['clamp_mode'],
            noise_std=kwargs['nerf_noise'],
            pred_occ=pred_occ
        )
        pixels = integ_results['rgb']
        depth = integ_results['depth']
        depth_var = integ_results['depth_var']
        weights = integ_results['weights']
        
        pixels = pixels.reshape((batch_size, img_size, img_size, 3))
        pixels = pixels.permute(0, 3, 1, 2).contiguous() * 2 - 1

        if with_normal_loss:
            on_surface_points = transformed_ray_origins.contiguous() + transformed_ray_directions.contiguous() * pred_depth.unsqueeze(2).expand(-1, -1, 3).contiguous()
            random_dirs = torch.rand_like(on_surface_points) - 0.5
            on_surface_points_neighbor = on_surface_points + random_dirs / random_dirs.norm(dim=-1, keepdim=True) * kwargs.get('h_sample', 1e-3)
            surface_points = torch.stack([on_surface_points, on_surface_points_neighbor], dim=2).reshape(batch_size, -1, 3)
            pred_normals = self.siren(surface_points, z, ray_directions=None, get_normal=True).reshape(batch_size, img_size * img_size, 2, 3)
            outputs['pred_normals'] = pred_normals

        if with_opacprior and pred_occ:
            integrated_opac = torch.sigmoid(all_outputs[..., 0])
            outputs['opacprior'] = torch.mean(
                torch.log(0.1 + integrated_opac.view(integrated_opac.size(0), -1)) + 
                torch.log(0.1 + 1. - integrated_opac.view(integrated_opac.size(0), -1)) + 2.20727
            )

        # observe variables
        outputs['occupancy'] = torch.sigmoid(all_outputs[..., -1].float().detach()).cpu()
        outputs['depth_map'] = depth.detach().cpu().clamp(min=ray_start, max=ray_end)
        outputs['depth_var'] = depth_var.detach().cpu()

        outputs['imgs'] = pixels
        outputs['pos'] = torch.cat([pitch, yaw], -1)

        return outputs


    def generate_avg_frequencies(self):
        """Calculates average frequencies and phase shifts"""

        z = torch.randn((10000, self.z_dim), device=self.siren.device)
        with torch.no_grad():
            frequencies, phase_shifts = self.siren.mapping_network(z)
        self.avg_frequencies = frequencies.mean(0, keepdim=True)
        self.avg_phase_shifts = phase_shifts.mean(0, keepdim=True)
        return self.avg_frequencies, self.avg_phase_shifts


    def staged_forward(
        self,
        z,
        img_size,
        fov,
        ray_start,
        ray_end,
        h_stddev,
        v_stddev,
        h_mean,
        v_mean,
        psi=1,
        lock_view_dependence=False,
        max_batch_size=50000,
        depth_map=False,
        sample_dist=None,
        hierarchical_sample=False,
        **kwargs
    ):
        """
        Similar to forward but used for inference.
        Calls the model sequencially using max_batch_size to limit memory usage.
        """

        batch_size = z.shape[0]
        outputs = {}

        self.generate_avg_frequencies()

        all_outputs = []
        all_z_vals = []

        # short hand
        pred_occ = kwargs.get('pred_occ', False)
        surface_sample = kwargs.get('surface_sample', False)
        num_steps_coarse = kwargs.get('num_steps_coarse', 9)
        num_steps_surface = kwargs.get('num_steps_surface', 9)
        num_steps_fine = kwargs.get('num_steps_fine', 6)

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

            if not surface_sample:
                all_outputs = [coarse_output]
                all_z_vals = [z_vals]
            else:
                # surface_z_vals shape: [batch_size, num_pixels**2, num_steps_surface, 1]
                surface_z_vals, pred_depth, mask = self.perform_ray_marching(
                    coarse_output.detach(),
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
                surface_z_vals = surface_z_vals.detach().to(self.device)
                surface_around_points = transformed_ray_origins.unsqueeze(2).contiguous() + transformed_ray_directions.unsqueeze(2).contiguous() * surface_z_vals.expand(-1,-1,-1,3).contiguous()
                surface_around_points, surface_z_vals = perturb_points(surface_around_points, surface_z_vals, transformed_ray_directions, self.device)
                surface_around_points = surface_around_points.reshape(batch_size, -1, 3)
                outputs['pred_depth'] = pred_depth.reshape(batch_size, img_size, img_size).contiguous().cpu().clamp(min=ray_start, max=ray_end)
                outputs['mask'] = mask.reshape(batch_size, img_size, img_size).contiguous().cpu()

                transformed_ray_directions_surface = self._expand_ray_directions(transformed_ray_directions, num_steps_surface, lock_view_dependence)
                # Sequentially evaluate siren with max_batch_size to avoid OOM
                surface_output = torch.zeros((batch_size, surface_around_points.shape[1], 4), device=self.device)
                for b in range(batch_size):
                    head = 0
                    while head < surface_around_points.shape[1]:
                        tail = head + max_batch_size
                        surface_output[b:b+1, head:tail] = self.siren.forward_with_frequencies_phase_shifts(
                            surface_around_points[b:b+1, head:tail], truncated_frequencies[b:b+1], truncated_phase_shifts[b:b+1], ray_directions=transformed_ray_directions_surface[b:b+1, head:tail])
                        head += max_batch_size

                surface_output = surface_output.reshape(batch_size, img_size * img_size, -1, 4)
            
                all_outputs.append(surface_output)
                all_z_vals.append(surface_z_vals)

            if hierarchical_sample:
                transformed_points = transformed_points.reshape(batch_size, img_size * img_size, num_steps_coarse, 3)
                weights = fancy_integration(
                    coarse_output,
                    z_vals,
                    device=self.device,
                    clamp_mode=kwargs['clamp_mode'],
                    noise_std=kwargs['nerf_noise'],
                    pred_occ=pred_occ
                )['weights']

                weights = weights.reshape(batch_size * img_size * img_size, num_steps_coarse) + 1e-5

                #### Start new importance sampling
                z_vals = z_vals.reshape(batch_size * img_size * img_size, num_steps_coarse)
                z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:])
                z_vals = z_vals.reshape(batch_size, img_size * img_size, num_steps_coarse, 1)
                fine_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1], num_steps_fine, det=False).detach().to(self.device)
                fine_z_vals = fine_z_vals.reshape(batch_size, img_size * img_size, num_steps_fine, 1)

                fine_points = transformed_ray_origins.unsqueeze(2).contiguous() + transformed_ray_directions.unsqueeze(2).contiguous() * fine_z_vals.expand(-1,-1,-1,3).contiguous()
                fine_points = fine_points.reshape(batch_size, img_size*img_size*num_steps_fine, 3)
                #### end new importance sampling

                transformed_ray_directions_fine = self._expand_ray_directions(transformed_ray_directions, num_steps_fine, lock_view_dependence)
                # Sequentially evaluate siren with max_batch_size to avoid OOM
                fine_output = torch.zeros((batch_size, fine_points.shape[1], 4), device=self.device)
                for b in range(batch_size):
                    head = 0
                    while head < fine_points.shape[1]:
                        tail = head + max_batch_size
                        fine_output[b:b+1, head:tail] = self.siren.forward_with_frequencies_phase_shifts(
                            fine_points[b:b+1, head:tail], truncated_frequencies[b:b+1], truncated_phase_shifts[b:b+1], ray_directions=transformed_ray_directions_fine[b:b+1, head:tail])
                        head += max_batch_size

                fine_output = fine_output.reshape(batch_size, img_size * img_size, num_steps_fine, 4)

                all_outputs.append(fine_output)
                all_z_vals.append(fine_z_vals)

            # combine all the outputs
            all_outputs = torch.cat(all_outputs, dim=-2)
            all_z_vals = torch.cat(all_z_vals, dim=-2)
            _, indices = torch.sort(all_z_vals, dim=-2)
            all_z_vals = torch.gather(all_z_vals, -2, indices)
            all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, 4))

            integ_results = fancy_integration(
                all_outputs,
                all_z_vals,
                device=self.device,
                white_back=kwargs.get('white_back', False),
                clamp_mode = kwargs['clamp_mode'],
                last_back=kwargs.get('last_back', False),
                fill_mode=kwargs.get('fill_mode', None),
                noise_std=kwargs['nerf_noise'],
                pred_occ=pred_occ,
            )
            pixels = integ_results['rgb']
            depth = integ_results['depth']
            depth_var = integ_results['depth_var']
            weights = integ_results['weights']

            outputs['occupancy'] = torch.sigmoid(all_outputs[..., -1].float().detach())
            outputs['depth_var'] = depth_var
            pixels = pixels.reshape((batch_size, img_size, img_size, 3))
            pixels = pixels.permute(0, 3, 1, 2).contiguous().cpu() * 2 - 1
            outputs['imgs'] = pixels
            depth_map = depth.reshape(batch_size, img_size, img_size).contiguous().cpu()
            outputs['depth_map'] = depth_map.clamp(min=ray_start, max=ray_end)
            outputs['weights'] = weights

            if kwargs.get('get_normal_with_pred_depth', False) and mask.sum() > 0:
                on_surface_points = transformed_ray_origins.contiguous() + transformed_ray_directions.contiguous() * pred_depth.unsqueeze(2).expand(-1, -1, 3).contiguous()
            else:
                on_surface_points = transformed_ray_origins.contiguous() + transformed_ray_directions.contiguous() * depth.expand(-1, -1, 3).contiguous()
            on_surface_points = on_surface_points.unsqueeze(2).reshape(batch_size, -1, 3)
            pred_normals = self.siren.forward_with_frequencies_phase_shifts(on_surface_points, truncated_frequencies, truncated_phase_shifts, ray_directions=transformed_ray_directions, get_normal=True)
            on_surface_normals = pred_normals.reshape(batch_size, img_size, img_size, 3).permute(0, 3, 1, 2)
            outputs['surface_normals'] = on_surface_normals / 2 + 0.5

            pred_rgb_sigma = self.siren.forward_with_frequencies_phase_shifts(on_surface_points, truncated_frequencies, truncated_phase_shifts, ray_directions=transformed_ray_directions)
            pred_rgb_sigma = pred_rgb_sigma.reshape(batch_size, img_size, img_size, 4)
            pred_rgb = pred_rgb_sigma[..., :3].float().permute(0, 3, 1, 2).cpu()
            outputs['single_imgs'] = pred_rgb

        torch.cuda.empty_cache() # clear the middle cache to free memory

        return outputs

    # Used for rendering interpolations
    def staged_forward_with_frequencies(
        self,
        truncated_frequencies,
        truncated_phase_shifts,
        img_size,
        fov,
        ray_start,
        ray_end,
        h_stddev,
        v_stddev,
        h_mean,
        v_mean,
        psi=0.7,
        lock_view_dependence=False,
        max_batch_size=50000,
        depth_map=False,
        sample_dist=None,
        hierarchical_sample=False,
        pose=None,
        **kwargs
    ):
        batch_size = truncated_frequencies.shape[0]
        outputs = {}

        all_outputs = []
        all_z_vals = []

        # short hand
        pred_occ = kwargs.get('pred_occ', False)
        surface_sample = kwargs.get('surface_sample', False)
        num_steps_coarse = kwargs.get('num_steps_coarse', 9)
        num_steps_surface = kwargs.get('num_steps_surface', 9)
        num_steps_fine = kwargs.get('num_steps_fine', 6)

        with torch.no_grad():
            points_cam, z_vals, rays_d_cam = get_initial_rays_trig(
                batch_size, num_steps_coarse, resolution=(img_size, img_size), device=self.device, fov=fov, ray_start=ray_start, ray_end=ray_end) # batch_size, pixels, num_steps_coarse, 1
            if pose is not None:
                pitch, yaw = pose[:,:1], pose[:,1:2]
                camera_origin = pose2origin(self.device, pitch, yaw, batch_size, 1)
                transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins = transform_sampled_points_camera_pose(points_cam, z_vals, rays_d_cam, camera_origin, pitch, yaw, self.device)
            else:
                transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw = transform_sampled_points(
                    points_cam, z_vals, rays_d_cam, h_stddev=h_stddev, v_stddev=v_stddev, h_mean=h_mean, v_mean=v_mean, device=self.device, mode=sample_dist)
            transformed_points = transformed_points.reshape(batch_size, img_size*img_size*num_steps_coarse, 3)

            transformed_ray_directions_coarse = self._expand_ray_directions(transformed_ray_directions, num_steps_coarse, lock_view_dependence)
            # BATCHED SAMPLE
            coarse_output = torch.zeros((batch_size, transformed_points.shape[1], 4), device=self.device)
            for b in range(batch_size):
                head = 0
                while head < transformed_points.shape[1]:
                    tail = head + max_batch_size
                    coarse_output[b:b+1, head:tail] = self.siren.forward_with_frequencies_phase_shifts(
                        transformed_points[b:b+1, head:tail], truncated_frequencies[b:b+1], truncated_phase_shifts[b:b+1], ray_directions=transformed_ray_directions_coarse[b:b+1, head:tail])
                    head += max_batch_size

            coarse_output = coarse_output.reshape(batch_size, img_size * img_size, num_steps_coarse, 4)
            # END BATCHED SAMPLE

            if not surface_sample:
                all_outputs = [coarse_output]
                all_z_vals = [z_vals]
            else:
                # surface_z_vals shape: [batch_size, num_pixels**2, num_steps_surface, 1]
                surface_z_vals, pred_depth, mask = self.perform_ray_marching(
                    coarse_output.detach(),
                    z_vals,
                    transformed_ray_origins,
                    transformed_ray_directions,
                    n_steps=num_steps_coarse,
                    n_samples=num_steps_surface,
                    interval=kwargs['interval'],
                    n_secant_steps=kwargs.get('n_secant_steps', 8),
                    tau=kwargs.get('tau', 0.5),
                    truncated_frequencies=truncated_frequencies,
                    truncated_phase_shifts=truncated_phase_shifts
                )
                surface_z_vals = surface_z_vals.detach().to(self.device)
                surface_around_points = transformed_ray_origins.unsqueeze(2).contiguous() + transformed_ray_directions.unsqueeze(2).contiguous() * surface_z_vals.expand(-1,-1,-1,3).contiguous()
                surface_around_points, surface_z_vals = perturb_points(surface_around_points, surface_z_vals, transformed_ray_directions, self.device)
                surface_around_points = surface_around_points.reshape(batch_size, -1, 3)
                outputs['pred_depth'] = pred_depth.reshape(batch_size, img_size, img_size).contiguous().cpu().clamp(min=ray_start, max=ray_end)
                outputs['mask'] = mask.reshape(batch_size, img_size, img_size).contiguous().cpu()

                transformed_ray_directions_surface = self._expand_ray_directions(transformed_ray_directions, num_steps_surface, lock_view_dependence)
                # BATCHED SAMPLE
                surface_output = torch.zeros((batch_size, surface_around_points.shape[1], 4), device=self.device)
                for b in range(batch_size):
                    head = 0
                    while head < surface_around_points.shape[1]:
                        tail = head + max_batch_size
                        surface_output[b:b+1, head:tail] = self.siren.forward_with_frequencies_phase_shifts(
                            surface_around_points[b:b+1, head:tail], truncated_frequencies[b:b+1], truncated_phase_shifts[b:b+1], ray_directions=transformed_ray_directions_surface[b:b+1, head:tail])
                        head += max_batch_size

                surface_output = surface_output.reshape(batch_size, img_size * img_size, -1, 4)

                all_outputs.append(surface_output)
                all_z_vals.append(surface_z_vals)

            if hierarchical_sample:
                transformed_points = transformed_points.reshape(batch_size, img_size * img_size, num_steps_coarse, 3)
                weights = fancy_integration(
                    coarse_output,
                    z_vals,
                    device=self.device,
                    clamp_mode=kwargs['clamp_mode'],
                    noise_std=kwargs['nerf_noise'],
                    pred_occ=pred_occ
                )['weights']

                weights = weights.reshape(batch_size * img_size * img_size, num_steps_coarse) + 1e-5
                z_vals = z_vals.reshape(batch_size * img_size * img_size, num_steps_coarse) # We squash the dimensions here. This means we importance sample for every batch for every ray
                z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
                z_vals = z_vals.reshape(batch_size, img_size * img_size, num_steps_coarse, 1)
                fine_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1],
                                    num_steps_fine, det=False).detach().to(self.device) # batch_size, num_pixels**2, num_steps_fine
                fine_z_vals = fine_z_vals.reshape(batch_size, img_size * img_size, num_steps_fine, 1)

                fine_points = transformed_ray_origins.unsqueeze(2).contiguous() + transformed_ray_directions.unsqueeze(2).contiguous() * fine_z_vals.expand(-1,-1,-1,3).contiguous() # dimensions here not matching
                fine_points = fine_points.reshape(batch_size, img_size*img_size*num_steps_fine, 3)
                #### end new importance sampling

                transformed_ray_directions_fine = self._expand_ray_directions(transformed_ray_directions, num_steps_fine, lock_view_dependence)
                # BATCHED SAMPLE
                fine_output = torch.zeros((batch_size, fine_points.shape[1], 4), device=self.device)
                for b in range(batch_size):
                    head = 0
                    while head < fine_points.shape[1]:
                        tail = head + max_batch_size
                        fine_output[b:b+1, head:tail] = self.siren.forward_with_frequencies_phase_shifts(
                            fine_points[b:b+1, head:tail], truncated_frequencies[b:b+1], truncated_phase_shifts[b:b+1], ray_directions=transformed_ray_directions_fine[b:b+1, head:tail])
                        head += max_batch_size

                fine_output = fine_output.reshape(batch_size, img_size * img_size, num_steps_fine, 4)
                # END BATCHED SAMPLE

                all_outputs.append(fine_output)
                all_z_vals.append(fine_z_vals)

            # combine all the outputs
            all_outputs = torch.cat(all_outputs, dim=-2)
            all_z_vals = torch.cat(all_z_vals, dim=-2)
            _, indices = torch.sort(all_z_vals, dim=-2)
            all_z_vals = torch.gather(all_z_vals, -2, indices)
            all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, 4))

            integ_results = fancy_integration(
                all_outputs,
                all_z_vals,
                device=self.device,
                white_back=kwargs.get('white_back', False),
                clamp_mode = kwargs['clamp_mode'],
                last_back=kwargs.get('last_back', False),
                fill_mode=kwargs.get('fill_mode', None),
                noise_std=kwargs['nerf_noise'],
                pred_occ=pred_occ,
            )
            pixels = integ_results['rgb']
            depth = integ_results['depth']
            depth_var = integ_results['depth_var']
            weights = integ_results['weights']

            outputs['depth_var'] = depth_var
            pixels = pixels.reshape((batch_size, img_size, img_size, 3))
            pixels = pixels.permute(0, 3, 1, 2).contiguous().cpu() * 2 - 1
            outputs['imgs'] = pixels
            depth_map = depth.reshape(batch_size, img_size, img_size).contiguous().cpu()
            outputs['depth_map'] = depth_map.clamp(min=ray_start, max=ray_end)

            if kwargs.get('get_normal_with_pred_depth', False) and mask.sum() > 0:
                on_surface_points = transformed_ray_origins.contiguous() + transformed_ray_directions.contiguous() * pred_depth.unsqueeze(2).expand(-1, -1, 3).contiguous()
            else:
                on_surface_points = transformed_ray_origins.contiguous() + transformed_ray_directions.contiguous() * depth.expand(-1, -1, 3).contiguous()
            on_surface_points = on_surface_points.unsqueeze(2).reshape(batch_size, -1, 3)
            pred_normals = self.siren.forward_with_frequencies_phase_shifts(on_surface_points, truncated_frequencies, truncated_phase_shifts, ray_directions=transformed_ray_directions, get_normal=True)
            on_surface_normals = pred_normals.reshape(batch_size, img_size, img_size, 3).permute(0, 3, 1, 2)
            outputs['surface_normals'] = on_surface_normals / 2 + 0.5

        torch.cuda.empty_cache()

        return outputs


    def forward_with_frequencies(
        self,
        frequencies,
        phase_shifts,
        img_size,
        fov,
        ray_start,
        ray_end,
        h_stddev,
        v_stddev,
        h_mean,
        v_mean,
        hierarchical_sample,
        sample_dist=None,
        lock_view_dependence=False,
        pose=None,
        **kwargs
    ):
        batch_size = frequencies.shape[0]
        outputs = {}

        all_outputs = []
        all_z_vals = []

        # short hand
        pred_occ = kwargs.get('pred_occ', False)
        surface_sample = kwargs.get('surface_sample', False)
        num_steps_coarse = kwargs.get('num_steps_coarse', 9)
        num_steps_surface = kwargs.get('num_steps_surface', 9)
        num_steps_fine = kwargs.get('num_steps_fine', 6)

        points_cam, z_vals, rays_d_cam = get_initial_rays_trig(
            batch_size, num_steps_coarse, resolution=(img_size, img_size), device=self.device, fov=fov, ray_start=ray_start, ray_end=ray_end)
        if pose is not None:
            pitch, yaw = pose[:,:1], pose[:,1:2]
            camera_origin = pose2origin(self.device, pitch, yaw, batch_size, 1)
            transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins = transform_sampled_points_camera_pose(points_cam, z_vals, rays_d_cam, camera_origin, pitch, yaw, self.device)
        else:
            transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw = transform_sampled_points(
                points_cam, z_vals, rays_d_cam, h_stddev=h_stddev, v_stddev=v_stddev, h_mean=h_mean, v_mean=v_mean, device=self.device, mode=sample_dist)
        transformed_points = transformed_points.reshape(batch_size, img_size*img_size*num_steps_coarse, 3)
            
        if not surface_sample:
            coarse_output = self.siren.forward_with_frequencies_phase_shifts(transformed_points, frequencies, phase_shifts, ray_directions=
                self._expand_ray_directions(transformed_ray_directions, num_steps_coarse, lock_view_dependence)).reshape(batch_size, img_size * img_size, num_steps_coarse, 4)
            all_outputs = [coarse_output]
            all_z_vals = [z_vals]
        else:
            with torch.no_grad():
                coarse_output = self.siren.forward_with_frequencies_phase_shifts(transformed_points, frequencies, phase_shifts, ray_directions=
                    self._expand_ray_directions(transformed_ray_directions, num_steps_coarse, lock_view_dependence)).reshape(batch_size, img_size * img_size, num_steps_coarse, 4)

                # surface_z_vals shape: [batch_size, num_pixels**2, num_steps_surface, 1]
                surface_z_vals, pred_depth, mask = self.perform_ray_marching(
                    coarse_output.detach(),
                    z_vals,
                    transformed_ray_origins,
                    transformed_ray_directions,
                    n_steps=num_steps_coarse,
                    n_samples=num_steps_surface,
                    interval=kwargs['interval'],
                    n_secant_steps=kwargs.get('n_secant_steps', 8),
                    tau=kwargs.get('tau', 0.5),
                    truncated_frequencies=frequencies,
                    truncated_phase_shifts=phase_shifts
                )
                surface_z_vals = surface_z_vals.detach()
                surface_around_points = transformed_ray_origins.unsqueeze(2).contiguous() + transformed_ray_directions.unsqueeze(2).contiguous() * surface_z_vals.expand(-1,-1,-1,3).contiguous()
                surface_around_points, surface_z_vals = perturb_points(surface_around_points, surface_z_vals, transformed_ray_directions, self.device)
                surface_around_points = surface_around_points.reshape(batch_size, -1, 3)
                outputs['mask'] = mask
                outputs['pred_depth'] = pred_depth.detach().cpu().clamp(min=ray_start, max=ray_end)

            surface_output = self.siren.forward_with_frequencies_phase_shifts(surface_around_points, frequencies, phase_shifts, ray_directions=
                self._expand_ray_directions(transformed_ray_directions, num_steps_surface, lock_view_dependence)).reshape(batch_size, img_size * img_size, -1, 4)

            all_outputs.append(surface_output)
            all_z_vals.append(surface_z_vals)

        if hierarchical_sample:
            with torch.no_grad():
                transformed_points = transformed_points.reshape(batch_size, img_size * img_size, num_steps_coarse, 3)
                weights = fancy_integration(
                    coarse_output,
                    z_vals,
                    device=self.device,
                    clamp_mode=kwargs['clamp_mode'],
                    noise_std=kwargs['nerf_noise'],
                    pred_occ=pred_occ
                )['weights']

                weights = weights.reshape(batch_size * img_size * img_size, num_steps_coarse) + 1e-5
                #### Start new importance sampling
                z_vals = z_vals.reshape(batch_size * img_size * img_size, num_steps_coarse) # We squash the dimensions here. This means we importance sample for every batch for every ray
                z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
                z_vals = z_vals.reshape(batch_size, img_size * img_size, num_steps_coarse, 1)
                fine_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1],
                                 num_steps_fine, det=False).detach() # batch_size, num_pixels**2, num_steps_fine
                fine_z_vals = fine_z_vals.reshape(batch_size, img_size * img_size, num_steps_fine, 1)

                fine_points = transformed_ray_origins.unsqueeze(2).contiguous() + transformed_ray_directions.unsqueeze(2).contiguous() * fine_z_vals.expand(-1,-1,-1,3).contiguous() # dimensions here not matching
                fine_points = fine_points.reshape(batch_size, img_size*img_size*num_steps_fine, 3)
                #### end new importance sampling

            fine_output = self.siren.forward_with_frequencies_phase_shifts(fine_points, frequencies, phase_shifts, ray_directions=
                self._expand_ray_directions(transformed_ray_directions, num_steps_fine, lock_view_dependence)).reshape(batch_size, img_size * img_size, -1, 4)

            # Combine course and fine points
            all_outputs.append(fine_output)
            all_z_vals.append(fine_z_vals)

        # combine all the outputs
        all_outputs = torch.cat(all_outputs, dim=-2)
        all_z_vals = torch.cat(all_z_vals, dim=-2)
        _, indices = torch.sort(all_z_vals, dim=-2)
        all_z_vals = torch.gather(all_z_vals, -2, indices)
        # Target sizes: [-1, -1, -1, 4].  Tensor sizes: [240, 512, 12]
        all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, 4))

        # Create images with NeRF
        integ_results = fancy_integration(
            all_outputs,
            all_z_vals,
            device=self.device,
            white_back=kwargs.get('white_back', False),
            last_back=kwargs.get('last_back', False),
            clamp_mode=kwargs['clamp_mode'],
            noise_std=kwargs['nerf_noise'],
            pred_occ=pred_occ
        )
        pixels = integ_results['rgb']
        depth = integ_results['depth']

        pixels = pixels.reshape((batch_size, img_size, img_size, 3))
        pixels = pixels.permute(0, 3, 1, 2).contiguous() * 2 - 1
        depth_map = depth.reshape(batch_size, img_size, img_size).contiguous().cpu()
        outputs['depth_map'] = depth_map.clamp(min=ray_start, max=ray_end)

        outputs['imgs'] = pixels
        outputs['pos'] = torch.cat([pitch, yaw], -1)

        return outputs

    def run_Bisection_method(
        self,
        z_low,
        z_high,
        n_secant_steps,
        ray_origins,
        ray_directions,
        z,
        logit_tau,
        tohalf=False,
        truncated_frequencies=None,
        truncated_phase_shifts=None
    ):
        ''' Runs the bisection method for interval [z_low, z_high].

        Args:
            z_low (tensor): start values for the interval
            z_high (tensor): end values for the interval
            n_secant_steps (int): number of steps
            ray_origins (tensor): ray start points
            ray_directions (tensor): ray direction vectors
            z (tensor): latent conditioned code z
            logit_tau (float): threshold value in probs
        '''
        z_pred = (z_low + z_high) / 2.
        for i in range(n_secant_steps):
            p_mid = ray_origins + z_pred.unsqueeze(-1) * ray_directions
            with torch.no_grad():
                if truncated_frequencies is not None and truncated_phase_shifts is not None:
                    rgb_sigma = self.siren.forward_with_frequencies_phase_shifts(p_mid.unsqueeze(1), truncated_frequencies, truncated_phase_shifts, ray_directions=ray_directions.unsqueeze(1))
                else:
                    rgb_sigma = self.siren(p_mid.unsqueeze(1), z, ray_directions=ray_directions.unsqueeze(1))
                if tohalf:
                    f_mid = rgb_sigma[..., -1].half().squeeze(1) - logit_tau
                else:
                    f_mid = rgb_sigma[..., -1].squeeze(1) - logit_tau
            inz_low = f_mid < 0
            z_low[inz_low] = z_pred[inz_low]
            z_high[inz_low == 0] = z_pred[inz_low == 0]
            z_pred = 0.5 * (z_low + z_high)

        return z_pred.data

    def run_Secant_method(
        self,
        f_low,
        f_high,
        z_low,
        z_high,
        n_secant_steps,
        ray_origins,
        ray_directions,
        z,
        logit_tau,
        tohalf=False,
        truncated_frequencies=None,
        truncated_phase_shifts=None
    ):
        ''' Runs the secant method for interval [z_low, z_high].

        Args:
            z_low (tensor): start values for the interval
            z_high (tensor): end values for the interval
            n_secant_steps (int): number of steps
            ray_origins (tensor): ray start points
            ray_directions (tensor): ray direction vectors
            z (tensor): latent conditioned code z
            logit_tau (float): threshold value in probs
        '''
        z_pred = - f_low * (z_high - z_low) / (f_high - f_low) + z_low
        for i in range(n_secant_steps):
            p_mid = ray_origins + z_pred.unsqueeze(-1) * ray_directions
            with torch.no_grad():
                if truncated_frequencies is not None and truncated_phase_shifts is not None:
                    rgb_sigma = self.siren.forward_with_frequencies_phase_shifts(p_mid.unsqueeze(1), truncated_frequencies, truncated_phase_shifts, ray_directions=ray_directions.unsqueeze(1))
                else:
                    rgb_sigma = self.siren(p_mid.unsqueeze(1), z, ray_directions=ray_directions.unsqueeze(1))
                if tohalf:
                    f_mid = rgb_sigma[..., -1].half().squeeze(1) - logit_tau
                else:
                    f_mid = rgb_sigma[..., -1].squeeze(1) - logit_tau
            inz_low = f_mid < 0
            if inz_low.sum() > 0:
                z_low[inz_low] = z_pred[inz_low]
                f_low[inz_low] = f_mid[inz_low]
            if (inz_low == 0).sum() > 0:
                z_high[inz_low == 0] = z_pred[inz_low == 0]
                f_high[inz_low == 0] = f_mid[inz_low == 0]

            z_pred = - f_low * (z_high - z_low) / (f_high - f_low) + z_low

        return z_pred.data

    def perform_ray_marching(
        self,
        rgb_sigma,
        z_vals,
        ray_origins,
        ray_directions,
        n_steps,
        n_samples,
        interval,
        z=None,
        tau=0.5,
        n_secant_steps=8,
        depth_range=[0.88, 1.12],
        method='secant',
        clamp_mode='relu',
        tohalf=False,
        truncated_frequencies=None,
        truncated_phase_shifts=None,
    ):
        ''' Performs ray marching to detect surface points.

        The function returns the surface points as well as z_i of the formula
            ray(z_i) = ray_origins + z_i * ray_directions
        which hit the surface points. In addition, masks are returned for
        illegal values.

        Args:
            rgb_sigma: the output of siren network
            ray_origins (tensor): ray start points of dimension B x N x 3
            ray_directions (tensor):ray direction vectors of dim B x N x 3
            interval: sampling interval
            z (tensor): latent conditioned code
            tau (float): threshold value
            n_secant_steps (int): number of secant refinement steps
            depth_range (tuple): range of possible depth values (not relevant when
                using cube intersection)
            method (string): refinement method (default: secant)
        '''
        # n_pts = W * H
        batch_size, n_pts, D = ray_origins.shape
        device = self.device
        if tohalf:
            logit_tau = torch.from_numpy(get_logits_from_prob(tau)[np.newaxis].astype(np.float16)).to(device)
        else:
            logit_tau = torch.from_numpy(get_logits_from_prob(tau)[np.newaxis].astype(np.float32)).to(device)
        
        alphas = rgb_sigma[..., -1] - logit_tau
        #print("alphas shape:", alphas.shape)
        
        # Create mask for valid points where the first point is not occupied
        mask_0_not_occupied = alphas[:, :, 0] < 0

        # Calculate if sign change occurred and concat 1 (no sign change) in
        # last dimension
        sign_matrix = torch.cat([torch.sign(alphas[:, :, :-1] * alphas[:, :, 1:]),
                                    torch.ones(batch_size, n_pts, 1).to(device)],
                                dim=-1)
        cost_matrix = sign_matrix * torch.arange(
            n_steps, 0, -1).float().to(device)
        # Get first sign change and mask for values where a.) a sign changed
        # occurred and b.) no a neg to pos sign change occurred (meaning from
        # inside surface to outside)
        values, indices = torch.min(cost_matrix, -1)
        mask_sign_change = values < 0
        mask_neg_to_pos = alphas[torch.arange(batch_size).unsqueeze(-1),
                                torch.arange(n_pts).unsqueeze(-0), indices] < 0

        # Define mask where a valid depth value is found
        mask = mask_sign_change & mask_neg_to_pos & mask_0_not_occupied

        # Get depth values and function values for the interval
        # to which we want to apply the Secant method
        n = batch_size * n_pts
        z_low = z_vals.view(n, n_steps, 1)[torch.arange(n), indices.view(n)].view(batch_size, n_pts)[mask]
        f_low = alphas.view(n, n_steps, 1)[torch.arange(n), indices.view(n)].view(batch_size, n_pts)[mask]
        indices = torch.clamp(indices + 1, max=n_steps-1)
        z_high = z_vals.view(n, n_steps, 1)[torch.arange(n), indices.view(n)].view(batch_size, n_pts)[mask]
        f_high = alphas.view(n, n_steps, 1)[torch.arange(n), indices.view(n)].view(batch_size, n_pts)[mask]
        
        ray_origins_masked = ray_origins[mask]
        ray_direction_masked = ray_directions[mask]

        # write z in pointwise format
        # TODO determine z.shape
        if z is not None and z.shape[-1] != 0:
            z = z.unsqueeze(1).repeat(1, n_pts, 1)[mask]
        if truncated_frequencies is not None and truncated_phase_shifts is not None:
            truncated_frequencies = truncated_frequencies.unsqueeze(1).repeat(1, n_pts, 1)[mask]
            truncated_phase_shifts = truncated_phase_shifts.unsqueeze(1).repeat(1, n_pts, 1)[mask]

        # Apply surface depth refinement step (e.g. Secant method)
        if method == 'secant' and mask.sum() > 0:
            d_pred = self.run_Secant_method(
                f_low.clone(), f_high.clone(), z_low.clone(), z_high.clone(), n_secant_steps, ray_origins_masked, ray_direction_masked, 
                z, logit_tau, tohalf=tohalf, truncated_frequencies=truncated_frequencies, truncated_phase_shifts=truncated_phase_shifts)
        elif method == 'bisection' and mask.sum() > 0:
            d_pred = self.run_Bisection_method(
                z_low.clone(), z_high.clone(), n_secant_steps, ray_origins_masked, ray_direction_masked, z, logit_tau, 
                tohalf=tohalf, truncated_frequencies=truncated_frequencies, truncated_phase_shifts=truncated_phase_shifts)
        else:
            d_pred = torch.ones(ray_direction_masked.shape[0]).to(device)

        # for sanity
        d_pred_out = torch.ones(batch_size, n_pts).to(device)
        d_pred_out[mask] = d_pred
        # sample points
        ray_start = torch.ones(batch_size, n_pts).to(device) * depth_range[0]
        ray_end = torch.ones(batch_size, n_pts).to(device) * depth_range[1]
        ray_start_masked = d_pred - interval
        ray_end_masked = d_pred + interval
        # in case of cross the near boundary
        mask_cross_near_bound = ray_start_masked < depth_range[0]
        ray_start_masked[mask_cross_near_bound] = depth_range[0]
        ray_end_masked[mask_cross_near_bound] = depth_range[0] + interval * 2
        # in case of cross the far boundary
        mask_cross_far_bound = ray_end_masked > depth_range[1]
        ray_end_masked[mask_cross_far_bound] = depth_range[1]
        ray_start_masked[mask_cross_far_bound] = depth_range[1] - interval * 2
        # for sanity
        ray_start[mask] = ray_start_masked
        ray_end[mask] = ray_end_masked

        
        # pred_z_vals shape: [B, n_pts, n_samples]
        z_vals_init = torch.linspace(0, 1, n_samples, device=device)
        pred_z_vals = ray_start.unsqueeze(-1) + (ray_end - ray_start).unsqueeze(-1) * z_vals_init

        return pred_z_vals.unsqueeze(-1), d_pred_out, mask
