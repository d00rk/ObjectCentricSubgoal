from PIL import Image
import os
import random
from diffusers.optimization import SchedulerType, Optimizer, TYPE_TO_SCHEDULER_FUNCTION
import numpy as np
import torch
from torchvision.utils import save_image

def to_uint8(x):
    x = torch.clamp(x, 0, 1).detach().cpu().numpy()
    x = (x * 255).astype(np.uint8)
    return x

@torch.no_grad()
def save_recon_grid(path, images, recons, masks, max_slots=6):
    """
    path [str]: path to save image.
    images: (B, H, W, 3) torch.float in [0, 1].
    recons: (B, H, W, 3) torch.float in [0, 1].
    masks: (B, S, H, W, 1) torch.float in [0, 1].
    """
    assert isinstance(images, torch.Tensor) and isinstance(recons, torch.Tensor) and isinstance(masks, torch.Tensor), "Args must be torch.Tensor."
    images = images.permute(0, 3, 1, 2).contiguous()        # (B, C, H, W)
    recons = recons.permute(0, 3, 1, 2).contiguous()        # (B, C, H, W)
    masks = masks.permute(0, 1, 4, 2, 3).contiguous()       # (B, S, 1, H, W)
    
    images = images.detach().cpu().float().clamp(0,1)
    recons = recons.detach().cpu().float().clamp(0,1)
    masks = masks.detach().cpu().float().clamp(0, 1)
    
    B, _, H, w = images.shape
    S = masks.size(1)
    if max_slots is not None:
        S = min(S, max_slots)
        masks = masks[:, :S]
        
    rows = []
    for b in range(B):
        row = [images[b], recons[b]]        # (3,H,W)
        m = masks[b]                        # (S,1,H,W)
        m = m.repeat(1, 3, 1, 1)            # (S,3,H,W)
        row += [m[i] for i in range(S)]
        row_cat = torch.cat(row, dim=-1)    # (3, H, W*(2+S))
        rows.append(row_cat)
    grid = torch.cat(rows, dim=-2)
    save_image(grid, path)
    
def save_ckpt(model, opt, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({'model': model.state_dict(),
                'opt': opt.state_dict()}, path)

def save_ckpt_state(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def get_scheduler(name, optimizer, num_warmup_steps=None, num_training_steps=None, **kwargs):
    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
    if name == SchedulerType.CONSTANT:
        return schedule_func(optimizer, **kwargs)
    
    if num_warmup_steps is None:
        raise ValueError(f'{name} requires "num_warmup_steps".')
    
    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, **kwargs)
    
    if num_training_steps is None:
        raise ValueError(f'{name} require "num_training_steps".')
    
    return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, **kwargs)
    