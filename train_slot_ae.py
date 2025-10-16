import os
import math
import json
from datetime import datetime
import random
import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from dataset.libero_dataset import LiberoDataset
from model.slot_autoencoder import SlotAttentionAutoEncoder
from utils.common import save_recon_grid, save_ckpt, set_seed, get_scheduler


def parse_amp_dtype(x):
    if isinstance(x, torch.dtype):
        return x
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ('float16', 'fp16', 'half'):
            return torch.float16
        if s in ('bfloat16', 'bf16'):
            return torch.bfloat16
    return torch.float16

def nchw_to_nhwc(x: torch.Tensor) -> torch.Tensor:
    # [B,C,H,W] -> [B,H,W,C]
    return x.permute(0, 2, 3, 1).contiguous()


def nhwc_to_nchw(x: torch.Tensor) -> torch.Tensor:
    # [B,H,W,C] -> [B,C,H,W]
    return x.permute(0, 3, 1, 2).contiguous()


@hydra.main(version_base=None, config_path='config', config_name='train_slot_ae')
def main(cfg: DictConfig):
    set_seed(cfg.train.seed)
    device = torch.device(cfg.train.device)
    
    dataset = LiberoDataset(seed=cfg.train.seed, **cfg.dataset)
    n = len(dataset)
    n_val = max(1, int(cfg.train.val_ratio * n))
    idx = list(range(n))
    random.shuffle(idx)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    
    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)
    
    train_loader = DataLoader(train_subset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.train.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_subset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers, pin_memory=True, drop_last=False)
    
    model = SlotAttentionAutoEncoder(**cfg.model)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr)
    lr_scheduler = get_scheduler(name=cfg.train.lr_scheduler, optimizer=optimizer, num_warmup_steps=cfg.train.lr_warmup_steps, num_training_steps=len(train_loader)*cfg.train.num_epochs)
    run_dir = cfg.logging.output_dir
    print(f'[Hydra run dir] {run_dir}')
    os.makedirs(f'{run_dir}/ckpt', exist_ok=True)
    os.makedirs(f'{run_dir}/visualization', exist_ok=True)
    os.makedirs(f'{run_dir}/logs', exist_ok=True)
    json_log_path = f'{run_dir}/logs/json_log.log'
    
    wandb_kwargs = dict(project=cfg.logging.project, dir=run_dir, config=OmegaConf.to_container(cfg, resolve=True), name=cfg.logging.name)
    wandb.init(**wandb_kwargs)
    wandb.summary['params_total'] = n_params
    wandb.define_metric('step')
    wandb.define_metric('train/*', step_metric='step')
    wandb.define_metric('validation/*', step_metric='step')
    wandb.define_metric('epoch')
    wandb.define_metric('lr')
        
    with open(json_log_path, 'a') as f:
        f.write(json.dumps({
            'event': 'run_start',
            'time': datetime.now().isoformat(),
            'config': OmegaConf.to_container(cfg, resolve=True)
        }))
        f.write('\n')
    
    step = 0
    best = math.inf
    use_amp = cfg.train.use_amp
    amp_dtype = parse_amp_dtype(cfg.train.amp_dtype)
    grad_clip = cfg.train.grad_clip
    use_scaler = use_amp and (amp_dtype is torch.float16) and cfg.train.use_grad_scaler
    scaler = GradScaler(enabled=use_scaler)
    for epoch in range(cfg.train.num_epochs):
        model.train()
        is_best = False
        log_dict = {}
        running = 0.0
        for batch in tqdm.tqdm(train_loader, desc=f'Train epoch: {epoch}', leave=False):
            img, meta = batch
            img = img.to(device)        # (B, H, W, C)
            with autocast(enabled=use_amp, dtype=amp_dtype):
                recon_combined, recons, masks, slots = model(img)
                loss = F.mse_loss(img, recon_combined)
                
            optimizer.zero_grad(set_to_none=True)
            if use_amp and amp_dtype is torch.float16 and scaler is not None:
                scaler.scale(loss).backward()
                if grad_clip is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            wandb.log({'step': step, 'epoch': epoch, 'train/loss': float(loss.item()), 'lr': lr_scheduler.get_last_lr()[0]}, step=step)
            running += loss.item() * img.size(0)
            lr_scheduler.step()
            step += 1
        train_loss = running / len(train_loader.dataset)
        log_dict.update({'epoch': epoch, 'train_loss':train_loss, 'lr': lr_scheduler.get_last_lr()[0]})
        
        if epoch % cfg.train.val_every == 0:
            with torch.no_grad():
                model.eval()
                running = 0.0
                for batch in tqdm.tqdm(val_loader, desc=f'Validation at epoch {epoch}', leave=False):
                    img, meta = batch
                    img = img.to(device)   
                    
                    with autocast(enabled=use_amp, dtype=amp_dtype):
                        recon_combined, recons, masks, slots = model(img)
                        loss = F.mse_loss(img, recon_combined)
                    
                    running += loss.item() * img.size(0)
                
                val_loss = running / len(val_loader.dataset)
                wandb.log({'step': step, 'epoch': epoch, 'validation/loss': float(val_loss)}, step=step)
                log_dict.update({'validation_loss': val_loss})
                
            if val_loss < best:
                save_ckpt(model, optimizer, os.path.join(run_dir, 'ckpt', f'{epoch:03d}_val_loss={val_loss:.3f}.pt'))
                best = val_loss
                is_best = True
        
        log_dict.update({'event': 'epoch_end', 'is_best': is_best})
        
        with open(json_log_path, 'a') as f:
            f.write(json.dumps(log_dict))
            f.write('\n')
        
        if epoch % cfg.train.vis_every == 0:
            model.eval()
            with torch.no_grad():
                b = next(iter(val_loader))
                image, meta = b
                image = image[:cfg.train.vis_n].to(device, non_blocking=True)           # (B, H, W, C)
                recon_combined, recons, masks, slots = model(image)                     # (B, H, W, C)
                save_recon_grid(os.path.join(run_dir, 'visualization', f'{epoch:03d}.png'), image, recon_combined, masks)
        
        if (epoch % cfg.train.ckpt_every == 0) and cfg.train.save_last_ckpt:
            save_ckpt(model, optimizer, os.path.join(run_dir, 'ckpt', f'latest.pt'))
            
    with open(json_log_path, 'a') as f:
        f.write(json.dumps({
            'event': 'run_end',
            'time': datetime.now().isoformat(),
            'best_valid': float(best)
        }))
    
    print(f'Train Ended at {datetime.now().isoformat()}. Results are saved at {run_dir}')
    

if __name__=='__main__':
    main()