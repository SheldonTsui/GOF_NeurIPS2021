import argparse
import numpy as np
import torch
from tqdm import tqdm

import curriculums
from generators.generators_weights import Generator_Weight
from generators import siren


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('generator_path', type=str)
    parser.add_argument('--seeds_start', type=int, default=0)
    parser.add_argument('--seeds_end', type=int, default=1000)
    parser.add_argument('--max_batch_size', type=int, default=2400000)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--curriculum', type=str, default='CelebA')
    opt = parser.parse_args()

    curriculum = getattr(curriculums, opt.curriculum)
    curriculum['num_steps_coarse'] = 36
    curriculum['num_steps_fine'] = 0 
    curriculum['num_steps_surface'] = 0 
    curriculum['img_size'] = opt.image_size
    curriculum['psi'] = 0.7
    curriculum['last_back'] = False
    curriculum['nerf_noise'] = 0
    curriculum = {key: value for key, value in curriculum.items() if type(key) is str}
    if 'interval_min' in curriculum:
        curriculum['interval'] = curriculum['interval_min']

    SIREN = getattr(siren, curriculum['model'])
    generator = Generator_Weight(SIREN, curriculum['latent_dim']).to(device)
    ema_file = opt.generator_path.split('generator')[0] + 'ema.pth'
    ema = torch.load(ema_file)
    ema.copy_to(generator.parameters())
    generator.set_device(device)
    generator.eval()
    
    var_list = []
    for seed in tqdm(range(opt.seeds_start, opt.seeds_end)):
        torch.manual_seed(seed)
        z = torch.randn((1, 256), device=device)
        with torch.no_grad():
            depth_var = generator.get_depth_var(z, **curriculum)

        # print('current depth var:', depth_var)
        var_list.append(depth_var)
        
    print('# samples, depth var:', len(var_list), np.mean(var_list) * 10000)
