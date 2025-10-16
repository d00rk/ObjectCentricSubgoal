import os
import math
from tqdm import tqdm
from datetime import datetime
import hydra
from omegaconf import DictConfig
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from dataset.libero_dataset import LiberoImageDataset
from policy.slot_dynamics import SlotDynamics
from utils.logging_utils import setup_wandb, json_logger
from utils.common import set_seed, get_scheduler, save_ckpt_state, save_recon_grid

def sinkhorn(log_alpha, iters=20, eps=1e-9):
    B, S, _ = log_alpha.shape
    log_u = torch.zeros(B, S, device=log_alpha.device)
    log_v = torch.zeros(B, S, device=log_alpha.device)
    for _ in range(iters):
        log_u = -torch.logsumexp(log_alpha + log_v.unsqueeze(1), dim=2)
        log_v = -torch.logsumexp(log_alpha + log_u.unsqueeze(2), dim=1)
    log_P = log_alpha + log_u.unsqueeze(2) + log_v.unsqueeze(1)
    return torch.exp(log_P)

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

def info_gain_loss(pred_err_seq, ent_seq, event_logits, topk, tau):
    B, T, S = pred_err_seq.shape
    def roll_err(e, shift):
        pad = torch.zeros_like(e[:, :abs(shift)])
        return torch.cat([e[:, abs(shift):], pad], dim=1) if shift < 0 else torch.cat([pad, e[:, :-shift]], dim=1)
    future = roll_err(pred_err_seq, -tau)
    ig = (pred_err_seq - future)
    ent_drop = F.pad(ent_seq[:,:-1] - ent_seq[:, 1:], (0,0,1,0))
    p = event_logits.log_softmax(dim=-1).exp()
    p_top = torch.topk(p, k=min(topk, S), dim=-1).values.mean(-1, keepdim=True)
    align = (p * (ig + ent_drop)).mean()
    sparsity = - (p_top.mean())
    return -align, sparsity

def psnr(pred, target):
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return torch.tensor(100.0)
    return 20 * torch.log10(1.0/torch.sqrt(mse))

def is_dist_available_and_initialized():
    return dist.is_available() and dist.is_initialized()

def get_world_size():
    return dist.get_world_size() if is_dist_available_and_initialized() else 1

def get_rank():
    return dist.get_rank() if is_dist_available_and_initialized() else 0

def is_main_process():
    return get_rank() == 0

def ddp_setup(backend='nccl'):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank, world_size, local_rank = 0, 1, 0
        os.environ["LOCAL_RANK"] = "0"
    
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, init_method='env://', rank=rank, world_size=world_size)
    dist.barrier()
    return local_rank

def unwrap_model(m):
    return m.module if isinstance(m, DDP) else m
        

@hydra.main(version_base=None, config_path='config', config_name='train_slot_dynamics')
def main(cfg: DictConfig):
    local_rank = ddp_setup(backend='nccl')
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    set_seed(cfg.train.seed + get_rank())
            
    train_dataset = LiberoImageDataset(seed=cfg.train.seed, mode='train', **cfg.dataset)
    val_dataset = LiberoImageDataset(seed=cfg.train.seed, mode='val', **cfg.dataset)
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=get_world_size(), rank=get_rank(), shuffle=True, drop_last=True)
    val_sampler = None
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers, pin_memory=True, drop_last=True,
                              sampler=train_sampler, shuffle=False, persistent_workers=cfg.train.num_workers>0)
    val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers, pin_memory=True, drop_last=False,
                            sampler=val_sampler, shuffle=False, persistent_workers=cfg.train.num_workers>0)
    
    model = SlotDynamics(**cfg.model)
    model = model.to(device)
    
    base = unwrap_model(model)
    
    find_unused = False
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=find_unused)
    
    params = list(model.parameters())
    optimizer = optim.AdamW(params, betas=tuple(cfg.optimizer.betas), weight_decay=cfg.optimizer.weight_decay)
    total_steps = cfg.train.num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(name=cfg.optimizer.lr_scheduler, optimizer=optimizer, num_warmup_steps=cfg.optimizer.lr_warmup_steps, num_training_steps=total_steps)

    if is_main_process():
        print(f"Params - AutoEncoder: {sum(p.numel() for p in base.autoencoder.parameters())/1e6:.3f}M, Dynamics: {sum(p.numel() for p in base.dynamics.parameters())/1e6:.3f}M")
        print(f"Params - Total: {sum(p.numel() for p in base.parameters())/1e6:.3f}M")
    
    run = None
    if is_main_process():
        output_dir = cfg.logging.output_dir
        print(f'[Hydra run dir] {output_dir}')
        os.makedirs(f'{output_dir}/ckpt', exist_ok=True)
        os.makedirs(f'{output_dir}/visualization', exist_ok=True)
        os.makedirs(f'{output_dir}/logs', exist_ok=True)
        json_log_path = f'{output_dir}/logs/json_log.log'
        run = setup_wandb(cfg)
        json_log = json_logger(json_log_path)
    else:
        def json_log(*args, **kwargs):
            pass
    
    global_step = 0
    best = math.inf
    use_amp = cfg.train.use_amp
    amp_dtype = parse_amp_dtype(cfg.train.amp_dtype)
    grad_clip = cfg.train.grad_clip
    use_scaler = use_amp and (amp_dtype is torch.float16) and cfg.train.use_grad_scaler
    scaler = GradScaler(enabled=use_scaler)
    
    max_train_steps = None
    max_train_epoch = None
    max_val_steps = None
    if cfg.train.debug:
        max_train_steps = 3
        max_train_epoch = 2
        max_val_steps = 3
        
    for epoch in range(cfg.train.num_epochs):
        train_sampler.set_epoch(epoch)
        
        model.train()
        
        log_dict = {}
        train_step = 0
        pbar = tqdm(train_loader, desc=f'Train epoch: {epoch}', leave=False)
        for batch in pbar:
            imgs = batch['agentview_rgb']
            imgs = imgs.to(device)      # (B, T, C, H, W)
            B, T, C, H, W = imgs.shape
            with autocast(enabled=use_amp, dtype=amp_dtype):
                preds, pred_slots, slots = model(imgs)
                
                # reconstruction loss for t+1
                loss_recon = F.mse_loss(preds, imgs)
                
                # temporal smoothness on slots
                loss_consist = (slots[:, 1:] - slots[:, :-1]).pow(2).mean()
                
                loss = cfg.loss.w_recon * loss_recon + cfg.loss.w_consist * loss_consist
            
            optimizer.zero_grad(set_to_none=True)
            if use_amp and amp_dtype is torch.float16 and scaler is not None:
                scaler.scale(loss).backward()
                if grad_clip is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(params, grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(params, grad_clip)
                optimizer.step()
            
            if is_main_process():
                log_dict = {
                    'epoch': epoch,
                    'step': global_step,
                    'lr': lr_scheduler.get_last_lr()[0],
                    'train/loss': float(loss.item()),
                    'train/loss_recon': float(loss_recon.item()), 
                    'train/loss_consist': float(loss_consist.item()),
                }
                run.log(log_dict)
                json_log(log_dict)
            
            if max_train_steps is not None and train_step >= max_train_steps:
                break
            
            lr_scheduler.step()
            train_step += 1
            global_step += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.3f}'})
        
        if is_dist_available_and_initialized():
            dist.barrier()
            
        if (epoch % cfg.train.val_every == 0) and is_main_process():
            model.eval()
            
            val_step = 0
            with torch.no_grad():
                val_losses, nval = 0.0, 0.0
                val_losses_recon, val_losses_consist = 0.0, 0.0
                psnr_recon, psnr_pred = [], []
                for batch in tqdm(val_loader, desc=f'Validation at epoch {epoch}', leave=False):
                    imgs = batch['agentview_rgb']
                    imgs = imgs.to(device)           # (B, T, C, H, W)
                    B, T, C, H, W = imgs.shape
                    
                    preds, pred_slots, slots = model(imgs)
                    
                    val_loss_recon = F.mse_loss(preds, imgs).item()
                    val_loss_consist = (slots[:, 1:] - slots[:, :-1]).pow(2).mean().item()
                    
                    val_loss = cfg.loss.w_recon * val_loss_recon + cfg.loss.w_consist * val_loss_consist
                    
                    val_losses += val_loss * B
                    val_losses_recon += val_loss_recon * B
                    val_losses_consist += val_loss_consist * B
                    nval += B
                    
                    psnr_recon.append(psnr(preds[:, 0], imgs[:, 0]).item())
                    psnr_pred.append(psnr(preds[:, 1], imgs[:, 1]).item())
                    
                    val_step += 1
                    if max_val_steps is not None and val_step >= max_val_steps:
                        break
                val_losses /= max(nval, 1)
                val_losses_recon /= max(nval, 1)
                val_losses_consist /= max(nval, 1)
                
                mean_psnr_recon = np.mean(psnr_recon)
                mean_psnr_pred = np.mean(psnr_pred)

            run.log({'epoch': epoch, 'step': global_step, 'validation/loss': val_losses, 'validation/loss_recon': val_losses_recon,
                     'validation/psnr_recon': mean_psnr_recon, 'validation/psnr_pred': mean_psnr_pred})
            json_log({'epoch': epoch, 'step': global_step, 'validation/loss': val_losses, 'validation/loss_recon': val_losses_recon,
                      'validation/psnr_recon': mean_psnr_recon, 'validation/psnr_pred': mean_psnr_pred})
            
            if val_losses < best:
                state = {
                    'epoch': epoch,
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': lr_scheduler.state_dict(),
                    'config': dict(cfg),
                }
                save_ckpt_state(state=state, path=os.path.join(output_dir, 'ckpt', f'{epoch:03d}_val_loss={val_loss:.3f}.pt'))
                best = val_loss
            
            del val_loss, val_losses
            del val_loss_recon, val_losses_recon
            del val_loss_consist, val_losses_consist
            del psnr_recon, mean_psnr_recon
            del psnr_pred, mean_psnr_pred
                
        if (epoch % cfg.train.vis_every == 0) and is_main_process():
            model.eval()
            with torch.no_grad():
                b = next(iter(val_loader))
                image = b['agentview_rgb']
                image = image[:cfg.train.vis_n].to(device, non_blocking=True)                                   # (B, T, C, H, W)
                recon_combined, recons, masks, slots = model.module.encode_and_decode(image)                    # recon_combined: (B, T, H, W, C) masks: (B, T, S, H, W, 1)
                image = image.permute(0, 1, 3, 4, 2).contiguous()                                               # (B, T, H, W, C)               
                save_recon_grid(os.path.join(output_dir, 'visualization', f'{epoch:03d}.png'), image[:, 0], recon_combined[:, 0], masks[:, 0])
            
        if (epoch % cfg.train.ckpt_every == 0) and cfg.train.save_last_ckpt and is_main_process():
            state = {
                'epoch': epoch,
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': lr_scheduler.state_dict(),
                'config': dict(cfg),
            }
            save_ckpt_state(state=state, path=os.path.join(output_dir, 'ckpt', f'latest.pt'))
        
        if max_train_epoch is not None and epoch >= max_train_epoch:
            break
        
        if epoch % 100 == 0:
            import gc
            gc.collect()
            
    if is_main_process():
        run.finish()
        print(f'Train finished at {datetime.now().isoformat()}. Results are save at {output_dir}.')
            

if __name__=='__main__':
    main()