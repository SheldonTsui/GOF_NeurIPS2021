import argparse
import os
import torch
from torchvision.utils import save_image
from tqdm import tqdm

import curriculums
from generators import generators, siren


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--seeds_start', type=int, default=0)
    parser.add_argument('--seeds_end', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='imgs')
    parser.add_argument('--max_batch_size', type=int, default=2400000)
    parser.add_argument('--lock_view_dependence', action='store_true')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--ray_step_multiplier', type=int, default=2)
    parser.add_argument('--curriculum', type=str, default='CelebA')
    opt = parser.parse_args()

    curriculum = getattr(curriculums, opt.curriculum)
    curriculum['num_steps_surface'] = curriculum['num_steps_surface'] * opt.ray_step_multiplier
    curriculum['num_steps_coarse'] = curriculum['num_steps_coarse'] * opt.ray_step_multiplier
    curriculum['num_steps_fine'] = curriculum['num_steps_fine'] * opt.ray_step_multiplier
    curriculum['img_size'] = opt.image_size
    curriculum['psi'] = 0.7
    curriculum['v_stddev'] = 0
    curriculum['h_stddev'] = 0
    curriculum['lock_view_dependence'] = opt.lock_view_dependence
    curriculum['last_back'] = curriculum.get('eval_last_back', False)
    curriculum['nerf_noise'] = 0
    curriculum = {key: value for key, value in curriculum.items() if type(key) is str}
    if 'interval_min' in curriculum:
        curriculum['interval'] = curriculum['interval_min']
    
    os.makedirs(opt.output_dir, exist_ok=True)

    SIREN = getattr(siren, curriculum['model'])
    generator = getattr(generators, curriculum['generator'])(SIREN, curriculum['latent_dim']).to(device)
    ema_file = opt.path.split('generator')[0] + 'ema.pth'
    ema = torch.load(ema_file)
    ema.copy_to(generator.parameters())
    generator.set_device(device)
    generator.eval()
    
    face_angles = [-0.5, -0.25, 0., 0.25, 0.5]

    face_angles = [a + curriculum['h_mean'] for a in face_angles]

    for seed in tqdm(range(opt.seeds_start, opt.seeds_end)):
        images = []
        for i, yaw in enumerate(face_angles):
            curriculum['h_mean'] = yaw
            torch.manual_seed(seed)
            z = torch.randn((1, 256), device=device)
            with torch.no_grad():
                outputs = generator.staged_forward(z, **curriculum)

            tensor_img = outputs['imgs'].detach()
            depth_map = outputs['depth_map'].detach()
            pred_depth = outputs.get('pred_depth', None)
            mask = outputs.get('mask', None)
            normals = outputs.get('surface_normals', None)
            single_img = outputs.get('single_imgs', None)
            save_image(tensor_img, os.path.join(opt.output_dir, f"img_{seed}_{yaw}_.png"), normalize=True)
            save_image(depth_map.reshape(curriculum['img_size'], curriculum['img_size']), os.path.join(opt.output_dir, f"depth_{seed}_{yaw}_.png"), normalize=True)
            if pred_depth is not None:
                save_image(pred_depth.reshape(curriculum['img_size'], curriculum['img_size']), os.path.join(opt.output_dir, f"pred_depth_{seed}_{yaw}_.png"), normalize=True)
            if mask is not None:
                save_image((mask*255.).reshape(curriculum['img_size'], curriculum['img_size']), os.path.join(opt.output_dir, f"mask_{seed}_{yaw}_.png"))
            if normals is not None:
                save_image(normals, os.path.join(opt.output_dir, f"normal_{seed}_{yaw}_.png"))
            if single_img is not None:
                save_image(single_img, os.path.join(opt.output_dir, f"single_{seed}_{yaw}_.png"), normalize=True)
            images.append(tensor_img)
        save_image(torch.cat(images), os.path.join(opt.output_dir, f'grid_{seed}.png'), normalize=True)
