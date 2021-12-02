"""Script to render a video using a trained pi-GAN  model."""

import argparse
import math
import os

from torchvision.utils import save_image

import torch
from tqdm import tqdm
import numpy as np
import curriculums
from generators import generators, siren

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('path', type=str)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--output_dir', type=str, default='vids')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--max_batch_size', type=int, default=2400000)
parser.add_argument('--lock_view_dependence', action='store_true')
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--ray_step_multiplier', type=int, default=2)
parser.add_argument('--num_frames', type=int, default=36)
parser.add_argument('--curriculum', type=str, default='CelebA')
opt = parser.parse_args()

os.makedirs(opt.output_dir, exist_ok=True)

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
curriculum['num_frames'] = opt.num_frames
curriculum['nerf_noise'] = 0
curriculum = {key: value for key, value in curriculum.items() if type(key) is str}
if 'interval_min' in curriculum:
    curriculum['interval'] = curriculum['interval_min']

def img_normalize(img):
    img = torch.clamp(img, min=-1, max=1)
    img = img / 2 + 0.5

    return img

def ellipse(i, num, x_max, y_max):
    theta = i / (num-1) * math.pi * 2
    yaw = math.cos(theta) * x_max + math.pi / 2
    pitch = math.sin(theta) * y_max + math.pi / 2
    return yaw, pitch

SIREN = getattr(siren, curriculum['model'])
generator = getattr(generators, curriculum['generator'])(SIREN, curriculum['latent_dim']).to(device)
ema_file = opt.path.split('generator')[0] + 'ema.pth'
ema = torch.load(ema_file)
ema.copy_to(generator.parameters())
generator.set_device(device)
generator.eval()

torch.manual_seed(opt.seed)
z = torch.randn((1, 256), device=device)
for i in tqdm(range(opt.num_frames)):
    yaw, pitch = ellipse(i, opt.num_frames, 0.5, 0.25)
    curriculum['h_mean'] = yaw
    curriculum['v_mean'] = pitch
    with torch.no_grad():
        outputs = generator.staged_forward(z, **curriculum)
    tensor_img = outputs['imgs']
    single_img = outputs['single_imgs']
    normals = outputs.get('surface_normals', None)
    
    save_image(img_normalize(tensor_img), os.path.join(opt.output_dir, f"img_{i:03d}.png"))
    save_image(single_img, os.path.join(opt.output_dir, f"single_img_{i:03d}.png"))
    if normals is not None:
        save_image(normals, os.path.join(opt.output_dir, f"normal_{i:03d}.png"))

# from images to video
# ffmpeg -r 15 -f image2 -i xxx.png -c:v libx264 -crf 25 -pix_fmt yuv420p xxx.mp4
