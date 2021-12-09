"""Train pi-GAN. Supports distributed training."""

import argparse
import os
import numpy as np
import math
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image

from generators import generators, siren
from discriminators import discriminators
import fid_evaluation

import datasets
import curriculums
from tqdm import tqdm
from datetime import datetime
import copy

from torch_ema import ExponentialMovingAverage

def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()

def load_images(images, curriculum, device):
    return_images = []
    head = 0
    for stage in curriculum['stages']:
        stage_images = images[head:head + stage['batch_size']]
        stage_images = F.interpolate(stage_images, size=stage['img_size'],  mode='bilinear', align_corners=True)
        return_images.append(stage_images)
        head += stage['batch_size']
    return return_images


def z_sampler(shape, device, dist):
    if dist == 'gaussian':
        z = torch.randn(shape, device=device)
    elif dist == 'uniform':
        z = torch.rand(shape, device=device) * 2 - 1
    return z

def _get_visualize_results(generator_, input_z, copied_metadata, output_dir, step, suffix):
    outputs = generator_.module.staged_forward(input_z,  **copied_metadata)
    gen_imgs = outputs['imgs']
    depth_map = outputs['depth_map']
    pred_depth = outputs.get('pred_depth', None)
    mask = outputs.get('mask', None)
    surface_normals = outputs.get('surface_normals', None)
    single_imgs = outputs.get('single_imgs', None)

    save_image(torch.clamp(gen_imgs[:25], min=-1, max=1) / 2 + 0.5, os.path.join(output_dir, f"{step}_{suffix}.png"), nrow=5)
    save_image(depth_map[:25].unsqueeze(1), os.path.join(output_dir, f"depth_{step}_{suffix}.png"), nrow=5, normalize=True)
    if pred_depth is not None:
        save_image(pred_depth[:25].unsqueeze(1), os.path.join(output_dir, f"pred_depth_{step}_{suffix}.png"), nrow=5, normalize=True)
    if mask is not None:
        save_image(mask[:25].unsqueeze(1) * 255., os.path.join(output_dir, f"mask_{step}_{suffix}.png"), nrow=5)
    if surface_normals is not None:
        save_image(surface_normals[:25], os.path.join(output_dir, f"normal_{step}_{suffix}.png"), nrow=5)
    if single_imgs is not None:
        save_image(single_imgs[:25], os.path.join(output_dir, f"single_{step}_{suffix}.png"), nrow=5)

def parse_print(print_variable):
    return print_variable.item() if type(print_variable) is torch.Tensor else print_variable

def train(rank, world_size, opt):
    torch.manual_seed(0)

    setup(rank, world_size, opt.port)
    torch.cuda.set_device(rank)
    device = torch.device(rank)

    curriculum = getattr(curriculums, opt.curriculum)
    metadata = curriculums.extract_metadata(curriculum, 0)

    fixed_z = z_sampler((25, 256), device='cpu', dist=metadata['z_dist'])

    SIREN = getattr(siren, metadata['model'])

    CHANNELS = 3

    if opt.use_logger and rank == 0:
        from tensorboardX import SummaryWriter
        logger = SummaryWriter(os.path.join(opt.output_dir, 'logs', datetime.now().strftime("%Y%m%d-%H%M%S")))
        variable_dir = os.path.join(opt.output_dir, 'variables')
        if not os.path.exists(variable_dir):
            os.mkdir(variable_dir)

    scaler = torch.cuda.amp.GradScaler()

    if opt.load_dir != '' and not opt.load_dict:
        generator = torch.load(os.path.join(opt.load_dir, 'generator.pth'), map_location=device)
        discriminator = torch.load(os.path.join(opt.load_dir, 'discriminator.pth'), map_location=device)
    else:
        generator = getattr(generators, metadata['generator'])(SIREN, metadata['latent_dim']).to(device)
        discriminator = getattr(discriminators, metadata['discriminator'])().to(device)
        ema = ExponentialMovingAverage(generator.parameters(), decay=0.999)
        ema2 = ExponentialMovingAverage(generator.parameters(), decay=0.9999)

    if opt.load_dir != '' and opt.load_dict:
        generator.load_state_dict(torch.load(os.path.join(opt.load_dir, 'generator.pth'), map_location=device), strict=False)
        discriminator.load_state_dict(torch.load(os.path.join(opt.load_dir, 'discriminator.pth'), map_location=device), strict=False)

    generator_ddp = DDP(generator, device_ids=[rank], find_unused_parameters=True)
    discriminator_ddp = DDP(discriminator, device_ids=[rank], find_unused_parameters=True, broadcast_buffers=False)
    generator = generator_ddp.module
    discriminator = discriminator_ddp.module


    if metadata.get('unique_lr', False):
        mapping_network_param_names = [name for name, _ in generator_ddp.module.siren.mapping_network.named_parameters()]
        mapping_network_parameters = [p for n, p in generator_ddp.named_parameters() if n in mapping_network_param_names]
        generator_parameters = [p for n, p in generator_ddp.named_parameters() if n not in mapping_network_param_names]
        optimizer_G = torch.optim.Adam([{'params': generator_parameters, 'name': 'generator'},
                                        {'params': mapping_network_parameters, 'name': 'mapping_network', 'lr':metadata['gen_lr']*0.2}],
                                       lr=metadata['gen_lr'], betas=metadata['betas'], weight_decay=metadata['weight_decay'])
    else:
        optimizer_G = torch.optim.Adam(generator_ddp.parameters(), lr=metadata['gen_lr'], betas=metadata['betas'], weight_decay=metadata['weight_decay'])

    optimizer_D = torch.optim.Adam(discriminator_ddp.parameters(), lr=metadata['disc_lr'], betas=metadata['betas'], weight_decay=metadata['weight_decay'])

    if opt.load_dir != '':
        ema = torch.load(os.path.join(opt.load_dir, 'ema.pth'), map_location=device)
        ema2 = torch.load(os.path.join(opt.load_dir, 'ema2.pth'), map_location=device)
        optimizer_G.load_state_dict(torch.load(os.path.join(opt.load_dir, 'optimizer_G.pth'), map_location=device))
        optimizer_D.load_state_dict(torch.load(os.path.join(opt.load_dir, 'optimizer_D.pth'), map_location=device))
        if not metadata.get('disable_scaler', False):
            scaler.load_state_dict(torch.load(os.path.join(opt.load_dir, 'scaler.pth'), map_location=device))

    generator_losses = []
    discriminator_losses = []

    if opt.set_step != None:
        generator.step = opt.set_step
        discriminator.step = opt.set_step

    if metadata.get('disable_scaler', False):
        scaler = torch.cuda.amp.GradScaler(enabled=False)

    generator.set_device(device)

    # ----------
    #  Training
    # ----------

    with open(os.path.join(opt.output_dir, 'options.txt'), 'w') as f:
        f.write(str(opt))
        f.write('\n\n')
        f.write(str(generator))
        f.write('\n\n')
        f.write(str(discriminator))
        f.write('\n\n')
        f.write(str(curriculum))

    torch.manual_seed(rank)
    dataloader = None
    total_progress_bar = tqdm(total = opt.n_epochs, desc = "Total progress", dynamic_ncols=True)
    total_progress_bar.update(discriminator.epoch)
    interior_step_bar = tqdm(dynamic_ncols=True)

    # record time
    if rank == 0:
        now = datetime.now()
        now = now.strftime("%d--%H:%M--")

    for _ in range (opt.n_epochs):
        total_progress_bar.update(1)

        metadata = curriculums.extract_metadata(curriculum, discriminator.step)

        # Set learning rates
        for param_group in optimizer_G.param_groups:
            if param_group.get('name', None) == 'mapping_network':
                param_group['lr'] = metadata['gen_lr'] * 5e-2
            else:
                param_group['lr'] = metadata['gen_lr']
            param_group['betas'] = metadata['betas']
            param_group['weight_decay'] = metadata['weight_decay']
        for param_group in optimizer_D.param_groups:
            param_group['lr'] = metadata['disc_lr']
            param_group['betas'] = metadata['betas']
            param_group['weight_decay'] = metadata['weight_decay']

        if not dataloader or dataloader.batch_size != metadata['batch_size']:
            dataloader, CHANNELS = datasets.get_dataset_distributed(metadata['dataset'],
                                        world_size,
                                        rank,
                                        **metadata)

            step_next_upsample = curriculums.next_upsample_step(curriculum, discriminator.step)
            step_last_upsample = curriculums.last_upsample_step(curriculum, discriminator.step)


            interior_step_bar.reset(total=(step_next_upsample - step_last_upsample))
            interior_step_bar.set_description(f"Progress to next stage")
            interior_step_bar.update((discriminator.step - step_last_upsample))

        for i, (imgs, _) in enumerate(dataloader):
            if discriminator.step % opt.model_save_interval == 0 and rank == 0:
                torch.save(ema, os.path.join(opt.output_dir, now + 'ema.pth'))
                torch.save(ema2, os.path.join(opt.output_dir, now + 'ema2.pth'))
                torch.save(generator_ddp.module.state_dict(), os.path.join(opt.output_dir, now + 'generator.pth'))
                torch.save(discriminator_ddp.module.state_dict(), os.path.join(opt.output_dir, now + 'discriminator.pth'))
                torch.save(optimizer_G.state_dict(), os.path.join(opt.output_dir, now + 'optimizer_G.pth'))
                torch.save(optimizer_D.state_dict(), os.path.join(opt.output_dir, now + 'optimizer_D.pth'))
                torch.save(scaler.state_dict(), os.path.join(opt.output_dir, now + 'scaler.pth'))
            metadata = curriculums.extract_metadata(curriculum, discriminator.step)

            if dataloader.batch_size != metadata['batch_size']: break

            if scaler.get_scale() < 1:
                scaler.update(1.)

            generator_ddp.train()
            discriminator_ddp.train()

            alpha = min(1, (discriminator.step - step_last_upsample) / (metadata['fade_steps']))

            real_imgs = imgs.to(device, non_blocking=True)

            metadata['nerf_noise'] = max(0, 1. - discriminator.step/5000.)
            if 'decay_beta' in metadata:
                metadata['interval'] = max(metadata['interval_min'], metadata['interval_max'] * np.exp(-metadata['decay_beta'] * generator.step))
            if metadata.get('with_opacprior', False):
                metadata['opacprior_lambda'] = min(metadata['opacprior_lambda_init'] * 10, metadata['opacprior_lambda_init'] * np.exp(metadata['opacprior_growth_rate'] * generator.step))

            # TRAIN DISCRIMINATOR
            with torch.cuda.amp.autocast():
                # Generate images for discriminator training
                with torch.no_grad():
                    z = z_sampler((real_imgs.shape[0], metadata['latent_dim']), device=device, dist=metadata['z_dist'])
                    split_batch_size = z.shape[0] // metadata['batch_split']
                    gen_imgs = []
                    gen_positions = []
                    for split in range(metadata['batch_split']):
                        subset_z = z[split * split_batch_size:(split+1) * split_batch_size]
                        outputs = generator_ddp(subset_z, **metadata)
                        g_imgs = outputs['imgs']
                        g_pos = outputs['pos']

                        gen_imgs.append(g_imgs)
                        gen_positions.append(g_pos)

                    gen_imgs = torch.cat(gen_imgs, axis=0)
                    gen_positions = torch.cat(gen_positions, axis=0)

                real_imgs.requires_grad = True
                r_preds, _, _ = discriminator_ddp(real_imgs, alpha, **metadata)

            if metadata['r1_lambda'] > 0:
                # Gradient penalty
                grad_real = torch.autograd.grad(outputs=scaler.scale(r_preds.sum()), inputs=real_imgs, create_graph=True)
                inv_scale = 1./scaler.get_scale()
                grad_real = [p * inv_scale for p in grad_real][0]
            with torch.cuda.amp.autocast():
                if metadata['r1_lambda'] > 0:
                    grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
                    grad_penalty = 0.5 * metadata['r1_lambda'] * grad_penalty
                else:
                    grad_penalty = 0

                g_preds, g_pred_latent, g_pred_position = discriminator_ddp(gen_imgs, alpha, **metadata)
                if metadata['z_lambda'] > 0 or metadata['pos_lambda'] > 0:
                    latent_penalty = torch.nn.MSELoss()(g_pred_latent, z) * metadata['z_lambda']
                    position_penalty = torch.nn.MSELoss()(g_pred_position, gen_positions) * metadata['pos_lambda']
                    identity_penalty = latent_penalty + position_penalty
                else:
                    identity_penalty = 0 

                d_loss = torch.nn.functional.softplus(g_preds).mean() + torch.nn.functional.softplus(-r_preds).mean() + grad_penalty + identity_penalty
                discriminator_losses.append(d_loss.item())

            optimizer_D.zero_grad()
            scaler.scale(d_loss).backward()
            scaler.unscale_(optimizer_D)
            torch.nn.utils.clip_grad_norm_(discriminator_ddp.parameters(), metadata['grad_clip'])
            scaler.step(optimizer_D)


            # TRAIN GENERATOR
            z = z_sampler((imgs.shape[0], metadata['latent_dim']), device=device, dist=metadata['z_dist'])

            split_batch_size = z.shape[0] // metadata['batch_split']

            for split in range(metadata['batch_split']):
                with torch.cuda.amp.autocast():
                    subset_z = z[split * split_batch_size:(split+1) * split_batch_size]
                    outputs = generator_ddp(subset_z, **metadata)
                    gen_imgs = outputs['imgs']
                    gen_positions = outputs['pos']
                    g_preds, g_pred_latent, g_pred_position = discriminator_ddp(gen_imgs, alpha, **metadata)

                    topk_percentage = max(0.99 ** (discriminator.step/metadata['topk_interval']), metadata['topk_v']) if 'topk_interval' in metadata and 'topk_v' in metadata else 1
                    topk_num = math.ceil(topk_percentage * g_preds.shape[0])

                    g_preds = torch.topk(g_preds, topk_num, dim=0).values

                    if metadata['z_lambda'] > 0 or metadata['pos_lambda'] > 0:
                        latent_penalty = torch.nn.MSELoss()(g_pred_latent, subset_z) * metadata['z_lambda']
                        position_penalty = torch.nn.MSELoss()(g_pred_position, gen_positions) * metadata['pos_lambda']
                        identity_penalty = latent_penalty + position_penalty
                    else:
                        identity_penalty = 0 

                    if metadata.get('with_normal_loss', False):
                        pred_normals = outputs.get('pred_normals', None)
                        mask = outputs.get('mask', None)
                        if (pred_normals is not None) and (mask is not None) and mask.sum() > 0:
                            surface_normals = pred_normals[..., 0][mask]
                            neighbor_normals = pred_normals[..., 1][mask]
                            normal_loss = torch.nn.MSELoss()(surface_normals, neighbor_normals) * metadata['normal_lambda']
                        else:
                            normal_loss = 0        
                    else:
                        normal_loss = 0 

                    tv_prior = outputs['tv_prior'] * metadata['tvprior_lambda'] if metadata.get('with_tvprior', False) else 0 
                    opacprior = outputs['opacprior'] * metadata['opacprior_lambda'] if metadata.get('with_opacprior', False) else 0 

                    g_loss = torch.nn.functional.softplus(-g_preds).mean() + identity_penalty + normal_loss + tv_prior + opacprior
                    generator_losses.append(g_loss.item())

                scaler.scale(g_loss).backward()

            scaler.unscale_(optimizer_G)
            torch.nn.utils.clip_grad_norm_(generator_ddp.parameters(), metadata.get('grad_clip', 0.3))
            scaler.step(optimizer_G)
            scaler.update()
            optimizer_G.zero_grad()
            ema.update(generator_ddp.parameters())
            ema2.update(generator_ddp.parameters())


            if rank == 0:
                interior_step_bar.update(1)
                if i%10 == 0:
                    print_msg = f"[Experiment: {opt.output_dir}] [GPU: {os.environ['CUDA_VISIBLE_DEVICES']}] [Epoch: {discriminator.epoch}/{opt.n_epochs}] "\
                        f"[D loss: {d_loss.item()}] [G loss: {g_loss.item()}] [identity penalty: {parse_print(identity_penalty)}] [normal loss: {parse_print(normal_loss)}] "\
                        f"[tv prior: {parse_print(tv_prior)}] [opacprior: {parse_print(opacprior)}] [Step: {discriminator.step}] [Alpha: {alpha:.2f}] "\
                        f"[Img Size: {metadata['img_size']}] [Batch Size: {metadata['batch_size']}] [TopK: {topk_num}] [Scale: {scaler.get_scale()}]\n"
                    tqdm.write(print_msg)

                if opt.use_logger and i%200 == 0:
                    logger.add_scalar('Loss/g_loss', g_loss, discriminator.step)
                    logger.add_scalar('Loss/d_loss', d_loss, discriminator.step)
                    logger.add_histogram('Depth/depth', outputs['depth_map'], discriminator.step)
                    logger.add_histogram('Depth/depth_var', outputs['depth_var'], discriminator.step)
                    logger.add_histogram('Depth/pred_depth', outputs['pred_depth'], discriminator.step)
                    logger.add_histogram('Occpancy', outputs['occupancy'], discriminator.step)

                if discriminator.step % opt.sample_interval == 0:
                    generator_ddp.eval()
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            copied_metadata = copy.deepcopy(metadata)
                            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                            copied_metadata['img_size'] = 128
                            _get_visualize_results(generator_ddp, fixed_z.to(device), copied_metadata, opt.output_dir, discriminator.step, suffix='fixed')
                            
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            copied_metadata = copy.deepcopy(metadata)
                            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                            copied_metadata['h_mean'] += 0.5
                            copied_metadata['img_size'] = 128
                            _get_visualize_results(generator_ddp, fixed_z.to(device), copied_metadata, opt.output_dir, discriminator.step, suffix='tilted')
                            
                    ema.store(generator_ddp.parameters())
                    ema.copy_to(generator_ddp.parameters())
                    generator_ddp.eval()
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            copied_metadata = copy.deepcopy(metadata)
                            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                            copied_metadata['img_size'] = 128
                            _get_visualize_results(generator_ddp, fixed_z.to(device), copied_metadata, opt.output_dir, discriminator.step, suffix='fixed_ema')
                            
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            copied_metadata = copy.deepcopy(metadata)
                            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                            copied_metadata['h_mean'] += 0.5
                            copied_metadata['img_size'] = 128
                            _get_visualize_results(generator_ddp, fixed_z.to(device), copied_metadata, opt.output_dir, discriminator.step, suffix='tilted_ema')
                            
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            copied_metadata = copy.deepcopy(metadata)
                            copied_metadata['img_size'] = 128
                            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                            copied_metadata['psi'] = 0.7
                            _get_visualize_results(generator_ddp, torch.randn_like(fixed_z).to(device), copied_metadata, opt.output_dir, discriminator.step, suffix='random')

                    ema.restore(generator_ddp.parameters())

                if discriminator.step % opt.sample_interval == 0:
                    torch.save(ema, os.path.join(opt.output_dir, 'ema.pth'))
                    torch.save(ema2, os.path.join(opt.output_dir, 'ema2.pth'))
                    torch.save(generator_ddp.module.state_dict(), os.path.join(opt.output_dir, 'generator.pth'))
                    torch.save(discriminator_ddp.module.state_dict(), os.path.join(opt.output_dir, 'discriminator.pth'))
                    torch.save(optimizer_G.state_dict(), os.path.join(opt.output_dir, 'optimizer_G.pth'))
                    torch.save(optimizer_D.state_dict(), os.path.join(opt.output_dir, 'optimizer_D.pth'))
                    torch.save(scaler.state_dict(), os.path.join(opt.output_dir, 'scaler.pth'))
                    torch.save(generator_losses, os.path.join(opt.output_dir, 'generator.losses'))
                    torch.save(discriminator_losses, os.path.join(opt.output_dir, 'discriminator.losses'))

            if opt.eval_freq > 0 and (discriminator.step + 1) % opt.eval_freq == 0:
                generated_dir = os.path.join(opt.output_dir, 'evaluation/generated')

                if rank == 0:
                    fid_evaluation.setup_evaluation(metadata['dataset'], generated_dir, target_size=128, num_imgs=metadata['num_eval_imgs'])
                dist.barrier()
                ema.store(generator_ddp.parameters())
                ema.copy_to(generator_ddp.parameters())
                generator_ddp.eval()
                fid_evaluation.output_images(generator_ddp, metadata, rank, world_size, generated_dir)
                ema.restore(generator_ddp.parameters())
                dist.barrier()
                if rank == 0:
                    fid = fid_evaluation.calculate_fid(metadata['dataset'], generated_dir, target_size=128)
                    with open(os.path.join(opt.output_dir, f'fid.txt'), 'a') as f:
                        now = datetime.now()
                        now = now.strftime("%d--%H:%M--")
                        f.write(f'\n{now} {discriminator.step}:{fid}')
                    if opt.use_logger:
                        logger.add_scalar('Loss/fid', fid, discriminator.step)

                torch.cuda.empty_cache()

            discriminator.step += 1
            generator.step += 1
        discriminator.epoch += 1
        generator.epoch += 1

    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=3000, help="number of epochs of training")
    parser.add_argument("--sample_interval", type=int, default=200, help="interval between image sampling")
    parser.add_argument('--output_dir', type=str, default='debug')
    parser.add_argument('--load_dir', type=str, default='')
    parser.add_argument('--load_dict', action='store_true')
    parser.add_argument('--use_logger', action='store_true')
    parser.add_argument('--curriculum', type=str, required=True)
    parser.add_argument('--eval_freq', type=int, default=5000)
    parser.add_argument('--port', type=str, default='12355')
    parser.add_argument('--set_step', type=int, default=None)
    parser.add_argument('--model_save_interval', type=int, default=5000)

    opt = parser.parse_args()
    print(opt)
    os.makedirs(opt.output_dir, exist_ok=True)
    num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    mp.spawn(train, args=(num_gpus, opt), nprocs=num_gpus, join=True)
