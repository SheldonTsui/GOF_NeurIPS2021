import argparse
import math
import os

from torchvision.utils import save_image

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import numpy as np
import curriculums
from generators import generators, siren

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('path', type=str)
parser.add_argument('--seeds', nargs='+', default=[0, 1, 2])
parser.add_argument('--output_dir', type=str, default='vids_interp')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--max_batch_size', type=int, default=1200000)
parser.add_argument('--depth_map', action='store_true')
parser.add_argument('--lock_view_dependence', action='store_true')
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--ray_step_multiplier', type=int, default=2)
parser.add_argument('--num_frames', type=int, default=36)
parser.add_argument('--curriculum', type=str, default='CelebA')
parser.add_argument('--trajectory', type=str, default='front')
parser.add_argument('--psi', type=float, default=0.7)
opt = parser.parse_args()

os.makedirs(opt.output_dir, exist_ok=True)

curriculum = getattr(curriculums, opt.curriculum)
curriculum['num_steps_surface'] = curriculum['num_steps_surface'] * opt.ray_step_multiplier
curriculum['num_steps_coarse'] = curriculum['num_steps_coarse'] * opt.ray_step_multiplier
curriculum['num_steps_fine'] = curriculum['num_steps_fine'] * opt.ray_step_multiplier
curriculum['img_size'] = opt.image_size
curriculum['psi'] = opt.psi
curriculum['v_stddev'] = 0
curriculum['h_stddev'] = 0
curriculum['lock_view_dependence'] = opt.lock_view_dependence
curriculum['last_back'] = curriculum.get('eval_last_back', False)
curriculum['num_frames'] = opt.num_frames
curriculum['nerf_noise'] = 0
curriculum = {key: value for key, value in curriculum.items() if type(key) is str}
if 'interval_min' in curriculum:
    curriculum['interval'] = curriculum['interval_min']

class FrequencyInterpolator:
    def __init__(self, generator, z1, z2, psi=0.5):
        avg_frequencies, avg_phase_shifts = generator.generate_avg_frequencies()
        raw_frequencies1, raw_phase_shifts1 = generator.siren.mapping_network(z1)
        self.truncated_frequencies1 = avg_frequencies + psi * (raw_frequencies1 - avg_frequencies)
        self.truncated_phase_shifts1 = avg_phase_shifts + psi * (raw_phase_shifts1 - avg_phase_shifts)
        raw_frequencies2, raw_phase_shifts2 = generator.siren.mapping_network(z2)
        self.truncated_frequencies2 = avg_frequencies + psi * (raw_frequencies2 - avg_frequencies)
        self.truncated_phase_shifts2 = avg_phase_shifts + psi * (raw_phase_shifts2 - avg_phase_shifts)

    def forward(self, t):
        frequencies = self.truncated_frequencies1 * (1-t) + self.truncated_frequencies2 * t
        phase_shifts = self.truncated_phase_shifts1 * (1-t) + self.truncated_phase_shifts2 * t

        return frequencies, phase_shifts

SIREN = getattr(siren, curriculum['model'])
generator = getattr(generators, curriculum['generator'])(SIREN, curriculum['latent_dim']).to(device)
ema_file = opt.path.split('generator')[0] + 'ema.pth'
ema = torch.load(ema_file)
ema.copy_to(generator.parameters())
generator.set_device(device)
generator.eval()

if opt.trajectory == 'front':
    trajectory = []
    for t in np.linspace(0, 1, curriculum['num_frames']):
        pitch = 0.2 * np.cos(t * 2 * math.pi) + math.pi/2
        yaw = 0.4 * np.sin(t * 2 * math.pi) + math.pi/2

        fov = curriculum['fov'] + 5 + np.sin(t * 2 * math.pi) * 5

        trajectory.append((t, pitch, yaw, fov))
elif opt.trajectory == 'orbit':
    trajectory = []
    for t in np.linspace(0, 1, curriculum['num_frames']):
        pitch = 0.2 * np.cos(t * 2 * math.pi) + math.pi/4
        yaw = t * 2 * math.pi
        fov = curriculum['fov']

        trajectory.append((t, pitch, yaw, fov))

print(opt.seeds)

for i, seed in enumerate(opt.seeds):
    torch.manual_seed(seed)
    z_current = torch.randn(1, 256, device=device)

    torch.manual_seed(opt.seeds[(i+1)%len(opt.seeds)])
    z_next = torch.randn(1, 256, device=device)

    frequencyInterpolator = FrequencyInterpolator(generator, z_current, z_next, psi=opt.psi)

    with torch.no_grad():
        im_idx = 0
        for t, pitch, yaw, fov in tqdm(trajectory):
            curriculum['h_mean'] = yaw
            curriculum['v_mean'] = pitch
            curriculum['fov'] = fov
            curriculum['h_stddev'] = 0
            curriculum['v_stddev'] = 0
            outputs = generator.staged_forward_with_frequencies(*frequencyInterpolator.forward(t), max_batch_size=opt.max_batch_size, depth_map=opt.depth_map, **curriculum)
                
            normals = outputs['surface_normals']
            frame = outputs['imgs']
            save_image(frame, os.path.join(opt.output_dir, f"img_{i:02d}_{im_idx:03d}.png"), normalize=True)
            save_image(normals, os.path.join(opt.output_dir, f"normal_{i:02d}_{im_idx:03d}.png"))
            im_idx += 1
