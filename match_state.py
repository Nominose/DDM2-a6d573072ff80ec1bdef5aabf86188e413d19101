import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
import os
import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt
from functools import partial
from scipy.stats import norm
import math
from tqdm import tqdm
import nibabel as nib

class Object:
    def __init__(self, config):
        self.config = config
        self.phase = 'train'
        self.gpu_ids = None
        self.debug = False

def _rev_warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_start * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[n_timestep - warmup_time:] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_16_128.json',
                    help='JSON file for configuration')
parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                    help='Run either train(training) or val(generation)', default='train')
parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
parser.add_argument('--debug', action='store_true')

# parse configs
args = parser.parse_args()

opt = Logger.parse(args)

# Convert to NoneDict, which return None for missing key.
opt = Logger.dict_to_nonedict(opt)
# logging
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

Logger.setup_logger(None, opt['path']['log'],
                    'train', level=logging.INFO, screen=True)
Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
logger = logging.getLogger('base')
logger.info('[Stage 2] Markov chain state matching (using teacher N2N)!')

# dataset
train_set = None
train_loader = None
val_set = None
val_loader = None

for phase, dataset_opt in opt['datasets'].items():
    dataset_opt['initial_stage_file'] = None
    if phase == 'train':
        # 强制加载所有 train samples
        dataset_opt['val_volume_idx'] = 'all'
        dataset_opt['val_slice_idx'] = 'all'
        train_set = Data.create_dataset(dataset_opt, phase)
        train_loader = Data.create_dataloader(
            train_set, dataset_opt, phase)
    elif phase == 'val':
        dataset_opt['val_volume_idx'] = 'all'
        dataset_opt['val_slice_idx'] = 'all'
        val_set = Data.create_dataset(dataset_opt, phase)
        val_loader = Data.create_dataloader(
            val_set, dataset_opt, phase)
logger.info('Initial Dataset Finished')

logger.info('Using teacher N2N results from ct_dataset')

#######
to_torch = partial(torch.tensor, dtype=torch.float32, device='cuda:0')

betas = _rev_warmup_beta(opt['noise_model']['beta_schedule']['linear_start'], opt['noise_model']['beta_schedule']['linear_end'],
                             1000, 0.7)

alphas = 1. - betas
alphas_cumprod = np.cumprod(alphas, axis=0)
alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
sqrt_alphas_cumprod_prev_np = np.sqrt(
            np.append(1., alphas_cumprod))
sqrt_alphas_cumprod_prev = to_torch(np.sqrt(
            np.append(1., alphas_cumprod)))


def compute_matched_state(data, sqrt_alphas_cumprod_prev):
    """Compute matched state for a single sample"""
    if 'denoised' not in data:
        return 500  # default value when no teacher result
    
    denoised = data['denoised'].cuda()
    
    min_lh = 999
    min_t = -1
    prev_diff = 999.
    
    for t in range(sqrt_alphas_cumprod_prev.shape[0]):
        noise = data['X'].cuda() - sqrt_alphas_cumprod_prev[t] * denoised
        noise_mean = torch.mean(noise)
        noise = noise - noise_mean

        mu, std = norm.fit(noise.cpu().numpy())

        diff = np.abs((1 - sqrt_alphas_cumprod_prev[t]**2).sqrt().cpu().numpy() - std)

        if diff < min_lh:
            min_lh = diff
            min_t = t

        if diff > prev_diff:
            break
        else:
            prev_diff = diff
    
    return min_t


# 打开文件写入
stage_file = open(opt['stage2_file'], 'w+')

# ========== 处理 train set ==========
if train_loader is not None:
    logger.info(f'Processing train set: {len(train_set.samples)} samples')
    idx = 0
    for _, data in tqdm(enumerate(train_loader), total=len(train_loader), desc='train'):
        volume_idx, slice_idx = train_set.samples[idx]
        idx += 1
        
        if 'denoised' not in data:
            stage_file.write('train_%d_%d_%d\n' % (volume_idx, slice_idx, 500))
            continue
        
        min_t = compute_matched_state(data, sqrt_alphas_cumprod_prev)
        
        if args.debug and idx == 30:
            denoised = data['denoised'].cuda()
            noise = torch.randn_like(denoised)
            result = sqrt_alphas_cumprod_prev[min_t] * denoised.detach() + (1. - sqrt_alphas_cumprod_prev[min_t]**2).sqrt() * noise
            denoised_np = denoised.detach().cpu().numpy()[0,0]
            input_np = data['X'].detach().cpu().numpy()[0,0]
            result_np = result.detach().cpu().numpy()[0,0]

            result_np = (result_np + 1.) / 2.
            input_np = (input_np + 1.) / 2.
            plt.imshow(np.hstack((input_np, result_np, denoised_np)), cmap='gray')
            plt.show()
            
            print(min_t, np.max(result_np), np.min(result_np))
            break
        
        stage_file.write('train_%d_%d_%d\n' % (volume_idx, slice_idx, min_t))

# ========== 处理 val set ==========
if val_loader is not None:
    logger.info(f'Processing val set: {len(val_set.samples)} samples')
    idx = 0
    for _, data in tqdm(enumerate(val_loader), total=len(val_loader), desc='val'):
        volume_idx, slice_idx = val_set.samples[idx]
        idx += 1
        
        if 'denoised' not in data:
            stage_file.write('val_%d_%d_%d\n' % (volume_idx, slice_idx, 500))
            continue
        
        min_t = compute_matched_state(data, sqrt_alphas_cumprod_prev)
        
        stage_file.write('val_%d_%d_%d\n' % (volume_idx, slice_idx, min_t))

stage_file.close()
print('done!')
