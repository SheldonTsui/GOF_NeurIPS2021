import os
import shutil
import torch
import math

from torch_fidelity import calculate_metrics
from torchvision.utils import save_image
from tqdm import tqdm
import copy
import argparse
import shutil

import curriculums
from generators import generators, siren

def main(generator_file, curriculum, num_images, max_batch_size, output_dir):
    SIREN = getattr(siren, curriculum['model'])
    generator = getattr(generators, curriculum['generator'])(SIREN, curriculum['latent_dim']).to(device)
    ema_file = generator_file.split('generator')[0] + 'ema.pth'
    ema = torch.load(ema_file)
    ema.copy_to(generator.parameters())
    generator.set_device(device)
    generator.eval()

    batch = 4
    for img_counter in tqdm(range(num_images//batch)):
        z = torch.randn(batch, 256, device=device)
        
        with torch.no_grad():
            outputs = generator.staged_forward(z, max_batch_size=max_batch_size, **curriculum)
            imgs = outputs['imgs'].detach()
            for i, img in enumerate(imgs):
                img_index = img_counter * batch + i
                save_image(img, os.path.join(output_dir, f'{img_index:0>5}.jpg'), normalize=True, range=(-1, 1))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('generator_path', type=str)
    parser.add_argument('--real_image_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='temp')
    parser.add_argument('--num_images', type=int, default=2048)
    parser.add_argument('--max_batch_size', type=int, default=94800000)
    parser.add_argument('--curriculum', type=str, default='CELEBA')
    parser.add_argument('--set_step', type=int, default=0)

    opt = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if os.path.exists(opt.output_dir) and os.path.isdir(opt.output_dir):
        shutil.rmtree(opt.output_dir)
    
    os.makedirs(opt.output_dir, exist_ok=False)
    
    curriculum = curriculums.extract_metadata(getattr(curriculums, opt.curriculum), opt.set_step)
    curriculum['img_size'] = 128
    curriculum['psi'] = 1
    curriculum['last_back'] = False
    curriculum['nerf_noise'] = 0

    if 'interval_min' in curriculum:
        curriculum['interval'] = curriculum['interval_min']

    main(opt.generator_path, curriculum, num_images=opt.num_images, max_batch_size=opt.max_batch_size, output_dir=opt.output_dir)
    metrics_dict = calculate_metrics(opt.output_dir, opt.real_image_dir, cuda=True, isc=True, fid=True, kid=True, verbose=False)
    print(metrics_dict)
    #with open('output/evalmetric/results.txt', 'a') as cur_file:
    #    cur_file.write(str(opt.generator_path))
    #    cur_file.write(str(metrics_dict) + '\n\n')
