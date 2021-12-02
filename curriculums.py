"""
To easily reproduce experiments, and avoid passing several command line arguments, we implemented
a curriculum utility. Parameters can be set in a curriculum dictionary.

Curriculum Schema:

    Numerical keys in the curriculum specify an upsample step. When the current step matches the upsample step,
    the values in the corresponding dict be updated in the curriculum. Common curriculum values specified at upsamples:
        batch_size: Batch Size.
        num_steps: Number of samples along ray.
        img_size: Generated image resolution.
        batch_split: Integer number over which to divide batches and aggregate sequentially. (Used due to memory constraints)
        gen_lr: Generator learnig rate.
        disc_lr: Discriminator learning rate.

    fov: Camera field of view
    ray_start: Near clipping for camera rays.
    ray_end: Far clipping for camera rays.
    fade_steps: Number of steps to fade in new layer on discriminator after upsample.
    h_stddev: Stddev of camera yaw in radians.
    v_stddev: Stddev of camera pitch in radians.
    h_mean:  Mean of camera yaw in radians.
    v_mean: Mean of camera yaw in radians.
    sample_dist: Type of camera pose distribution. (gaussian | spherical_uniform | uniform)
    topk_interval: Interval over which to fade the top k ratio.
    topk_v: Minimum fraction of a batch to keep during top k training.
    betas: Beta parameters for Adam.
    unique_lr: Whether to use reduced LRs for mapping network.
    weight_decay: Weight decay parameter.
    r1_lambda: R1 regularization parameter.
    latent_dim: Latent dim for Siren network  in generator.
    grad_clip: Grad clipping parameter.
    model: Siren architecture used in generator. (SPATIALSIRENBASELINE | TALLSIREN)
    generator: Generator class. (ImplicitGenerator3d)
    discriminator: Discriminator class. (ProgressiveEncoderDiscriminator | ProgressiveDiscriminator)
    dataset: Training dataset. (CelebA | Carla | Cats)
    clamp_mode: Clamping function for Siren density output. (relu | softplus)
    z_dist: Latent vector distributiion. (gaussian | uniform)
    hierarchical_sample: Flag to enable hierarchical_sampling from NeRF algorithm. (Doubles the number of sampled points)
    z_labmda: Weight for experimental latent code positional consistency loss.
    pos_lambda: Weight parameter for experimental positional consistency loss.
    last_back: Flag to fill in background color with last sampled color on ray.
"""

import math

def next_upsample_step(curriculum, current_step):
    # Return the epoch when it will next upsample
    current_metadata = extract_metadata(curriculum, current_step)
    current_size = current_metadata['img_size']
    for curriculum_step in sorted([cs for cs in curriculum.keys() if type(cs) == int]):
        if curriculum_step > current_step and curriculum[curriculum_step].get('img_size', 512) > current_size:
            return curriculum_step
    return float('Inf')

def last_upsample_step(curriculum, current_step):
    # Returns the start epoch of the current stage, i.e. the epoch
    # it last upsampled
    current_metadata = extract_metadata(curriculum, current_step)
    current_size = current_metadata['img_size']
    for curriculum_step in sorted([cs for cs in curriculum.keys() if type(cs) == int]):
        if curriculum_step <= current_step and curriculum[curriculum_step]['img_size'] == current_size:
            return curriculum_step
    return 0

def get_current_step(curriculum, epoch):
    step = 0
    for update_epoch in curriculum['update_epochs']:
        if epoch >= update_epoch:
            step += 1
    return step

def extract_metadata(curriculum, current_step):
    return_dict = {}
    for curriculum_step in sorted([cs for cs in curriculum.keys() if type(cs) == int], reverse=True):
        if curriculum_step <= current_step:
            for key, value in curriculum[curriculum_step].items():
                return_dict[key] = value
            break
    for key in [k for k in curriculum.keys() if type(k) != int]:
        return_dict[key] = curriculum[key]
    return return_dict


BFM = {
    0: {'batch_size': 16, 'img_size': 32, 'batch_split': 1, 'gen_lr': 5e-5, 'disc_lr': 2e-4},
    int(10e3): {'batch_size': 8, 'img_size': 64, 'batch_split': 1, 'gen_lr': 5e-5, 'disc_lr': 2e-4},
    int(60e3): {'batch_size': 4, 'img_size': 128, 'batch_split': 2, 'gen_lr': 4e-6, 'disc_lr': 2e-5},
    int(80e3): {},

    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
    'h_stddev': 0.3,
    'v_stddev': 0.155,
    'h_mean': math.pi*0.5,
    'v_mean': math.pi*0.5,
    'sample_dist': 'gaussian',
    'topk_interval': 2000,
    'topk_v': 0.6,
    'betas': (0, 0.9),
    'unique_lr': True,
    'weight_decay': 0,
    'r1_lambda': 1,
    'latent_dim': 256,
    'grad_clip': 1,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'ImplicitGenerator3d',
    'discriminator': 'ProgressiveEncoderDiscriminator',
    'dataset': 'BFM',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'num_eval_imgs': 8000,
    'pred_occ': True,
    'decay_beta': 3e-5,
    'interval_max': 0.12,
    'interval_min': 0.02,
    'surface_sample': True,
    'n_secant_steps': 4,
    'get_normal_with_pred_depth': True,
    'with_opacprior': True,
    'opacprior_lambda_init': 0.1,
    'opacprior_growth_rate': 4e-5,
    'with_normal_loss': True,
    'normal_lambda': 0.01,
    'h_sample': 1e-3,
    'tau': 0.5,
    'num_steps_surface': 12,
    'num_steps_coarse': 9,
    'num_steps_fine': 3,
}


CelebA = {
    0: {'batch_size': 16, 'img_size': 32, 'batch_split': 1, 'gen_lr': 5e-5, 'disc_lr': 2e-4},
    int(20e3): {'batch_size': 8, 'img_size': 64, 'batch_split': 1, 'gen_lr': 5e-5, 'disc_lr': 2e-4},
    int(160e3): {'batch_size': 4, 'img_size': 128, 'batch_split': 2, 'gen_lr': 4e-6, 'disc_lr': 2e-5},
    int(200e3): {},

    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
    'h_stddev': 0.3,
    'v_stddev': 0.155,
    'h_mean': math.pi*0.5,
    'v_mean': math.pi*0.5,
    'sample_dist': 'gaussian',
    'topk_interval': 2000,
    'topk_v': 0.6,
    'betas': (0, 0.9),
    'unique_lr': True,
    'weight_decay': 0,
    'r1_lambda': 1,
    'latent_dim': 256,
    'grad_clip': 1,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'ImplicitGenerator3d',
    'discriminator': 'ProgressiveEncoderDiscriminator',
    'dataset': 'CelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'num_eval_imgs': 8000,
    'pred_occ': True,
    'decay_beta': 1e-5,
    'interval_max': 0.12,
    'interval_min': 0.03,
    'surface_sample': True,
    'n_secant_steps': 4,
    'get_normal_with_pred_depth': True,
    'with_opacprior': True,
    'opacprior_lambda_init': 0.05,
    'opacprior_growth_rate': 0.5e-5,
    'with_normal_loss': True,
    'normal_lambda': 0.05,
    'h_sample': 0.01,
    'tau': 0.5,
    'num_steps_surface': 9,
    'num_steps_coarse': 9,
    'num_steps_fine': 6,
}


CATS = {
    0: {'batch_size': 16, 'img_size': 32, 'batch_split': 1, 'gen_lr': 4e-5, 'disc_lr': 2e-4},
    int(5e3): {'batch_size': 8, 'img_size': 64, 'batch_split': 1, 'gen_lr': 4e-5, 'disc_lr': 2e-4},
    int(30e3): {'batch_size': 4, 'img_size': 128, 'batch_split': 2, 'gen_lr': 4e-6, 'disc_lr': 2e-5},
    int(40e3): {},


    'fov': 12,
    'ray_start': 0.8,
    'ray_end': 1.2,
    'fade_steps': 10000,
    'h_stddev': 0.5,
    'v_stddev': 0.4,
    'h_mean': math.pi*0.5,
    'v_mean': math.pi*0.5,
    'sample_dist': 'uniform',
    'topk_interval': 2000,
    'topk_v': 0.6,
    'betas': (0, 0.9),
    'unique_lr': True,
    'weight_decay': 0,
    'r1_lambda': 1,
    'latent_dim': 256,
    'grad_clip': 1,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'ImplicitGenerator3d',
    'discriminator': 'ProgressiveEncoderDiscriminator',
    'dataset': 'Cats',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'num_eval_imgs': 6444,
    'pred_occ': True,
    'decay_beta': 2e-5,
    'interval_max': 0.2,
    'interval_min': 0.1,
    'surface_sample': True,
    'n_secant_steps': 4,
    'get_normal_with_pred_depth': True,
    'with_opacprior': True,
    'opacprior_lambda_init': 0.05,
    'opacprior_growth_rate': 1e-5,
    'with_normal_loss': True,
    'normal_lambda': 0.05,
    'h_sample': 0.01,
    'tau': 0.5,
    'num_steps_surface': 9,
    'num_steps_coarse': 9,
    'num_steps_fine': 6,
}