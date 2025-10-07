import os
import math
from tqdm import tqdm
from datetime import datetime
import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from dataset.libero_dataset import LiberoImageDataset
from model.slot_former import SlotEncoder, SlotFormer
from model.event_detector import EventDetector
from utils.surprise import *
from utils.logging_utils import setup_wandb, json_logger
from utils.common import set_seed, get_scheduler, save_ckpt_state

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

@hydra.main(version_base=None, config_path='config', config_name='train_subgoal_predictor')
def main(cfg: DictConfig):
    set_seed(cfg.train.seed)
    device = torch.device(cfg.train.device)
    
    views =[]
    sp = cfg.shape_meta
    for k, v in sp['obs'].items():
        t = v.get('type', 'low_dim')
        if t == 'rgb':
            views.append(k)
            
    train_dataset = LiberoImageDataset(seed=cfg.train.seed, mode='train', **cfg.dataset)
    val_dataset = LiberoImageDataset(seed=cfg.train.seed, mode='val', **cfg.dataset)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.train.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers, pin_memory=True, drop_last=False)
    
    enc = SlotEncoder(**cfg.model.encoder)
    enc = enc.to(device)
    
    dyn = SlotFormer(**cfg.model.dynamics)
    dyn = dyn.to(device)
    
    evt = EventDetector(**cfg.model.event_detector)
    evt = evt.to(device)
    
    params = list(enc.parameters()) + list(dyn.parameters()) + list(evt.parameters())
    optimizer = optim.AdamW(params, lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
    total_steps = cfg.train.num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(name=cfg.optimizer.lr_scheduler, optimizer=optimizer, num_warmup_steps=cfg.optimizer.lr_warmup_steps, num_training_steps=total_steps)
    
    print(f"Params - Encoder:{sum(p.numel() for p in enc.parameters() if p.requires_grad)/1e6:.3f}M, SlotFormer: {sum(p.numel() for p in dyn.parameters() if p.requires_grad)/1e6:.3f}M, Event Detector: {sum(p.numel() for p in evt.parameters() if p.requires_grad)/1e6:.3f}M")
    print(f"Params - Total: {sum(p.numel() for model in [enc, dyn, evt] for p in model.parameters())/1e6:.3f}M")
    
    output_dir = cfg.logging.output_dir
    print(f'[Hydra run dir] {output_dir}')
    os.makedirs(f'{output_dir}/ckpt', exist_ok=True)
    os.makedirs(f'{output_dir}/visualization', exist_ok=True)
    os.makedirs(f'{output_dir}/logs', exist_ok=True)
    json_log_path = f'{output_dir}/logs/json_log.log'
    
    run = setup_wandb(cfg)
    json_log = json_logger(json_log_path)
    
    ema_mean, ema_var = 0.0, 1.0
    alpha = 0.99
    
    vref = cfg.train.reference_view
    
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
        enc.train()
        dyn.train()
        evt.train()
        log_dict = {}
        running = 0.0
        train_step = 0
        pbar = tqdm(train_loader, desc=f'Train epoch: {epoch}', leave=False)
        for batch in pbar:
            imgs = []
            for v in views:
                img = batch[v]              # (B, T, C, H, W)
                B, T, C, H, W = img.shape
                imgs.append(img)
            imgs = torch.stack(imgs, dim=1).to(device)      # (B, V, T, C, H, W)
            
            with autocast(enabled=use_amp, dtype=amp_dtype):
                z_views = []
                for v in range(len(views)):
                    z_v = enc(imgs[:, v])           # (B, T, S, D)
                    z_views.append(z_v)     
                z = torch.stack(z_views, dim=1)     # (B, V, T, S, D)
                
                z_gt = z[:, :, -1]                  # (B, V, S, D)
                z_preds = []
                for v in range(len(views)):
                    z_pred = dyn(z[:, v, :-1])      # (B, S, D)
                    z_preds.append(z_pred)
                z_pred = torch.stack(z_preds, dim=1)    # (B, V, S, D)
                
                s = multiview_predictive_surprise(z_pred, z_gt, reduce='mean')
                dR = multiview_relational_change(z[:, :, -2], z_gt, reduce='mean')
                
                with torch.no_grad():
                    stat = (s + cfg.loss.rel_weight * dR).mean().detach()
                    ema_mean = alpha * ema_mean + (1-alpha) * stat.item()
                    ema_var = alpha * ema_var + (1-alpha) * ((s+cfg.loss.rel_weight * dR).var().detach().item() + 1e-6)
                    ema_std = math.sqrt(max(ema_var, 1e-6))
                
                Wwin = cfg.model.event_detector.window
                if T < Wwin:
                    pad = Wwin - T
                    z_win = torch.cat([z[:, :, :1].repeat(1, 1, pad, 1, 1), z], dim=2)[:, :, -Wwin:]
                else:
                    z_win = z[:, :, -Wwin:]
                z_win_fused = z_win.mean(dim=1)     # (B, W, S, D)
                logits_evt = evt(z_win_fused)
                y = soft_labels(s, dR, ema_mean, ema_std, cfg.loss.soft_label_temp, cfg.loss.rel_weight)
                
                L_dyn = (z_pred - z_gt).pow(2).mean()
                
                L_cv, L_rel = 0.0, 0.0
                
                if vref in views:
                    vref = views.index(vref)
                else:
                    vref = 0
                
                for v in range(len(views)):
                    if v == vref: continue
                    z1 = z[:, vref, -1]     # (B, S, D)
                    z2 = z[:, v, -1]        # (B, S, D)
                    cmat = torch.cdist(z1, z2, p=2) ** 2    # (B, S, S)
                    log_alpha = -cmat / max(cfg.loss.ot_temp, 1e-6)
                    p = sinkhorn(log_alpha, iters=cfg.loss.sinkhorn_iters)
                    z2_perm = torch.bmm(p, z2)
                    L_cv = L_cv + (z2_perm - z1).pow(2).mean()
                    def pdist(z):
                        diff = z.unsqueeze(2) - z.unsqueeze(1)
                        return diff.pow(2).sum(-1).sqrt()
                    d1 = pdist(z1)
                    d2 = pdist(z2)
                    d2_perm = torch.bmm(torch.bmm(p, d2), p.transpose(1,2))
                    L_rel = L_rel + (d2_perm - d1).pow(2).mean()
                
                if len(views) > 1:
                    L_cv = L_cv / (len(views) - 1)
                    L_rel = L_rel / (len(views) - 1)
                
                L_evt = F.binary_cross_entropy_with_logits(logits_evt, y)
                L_sparsity = cfg.loss.beta_sparsity * torch.sigmoid(logits_evt).mean()
                
                loss = L_dyn + cfg.loss.lambda_evt * L_evt + L_sparsity + cfg.loss.lambda_cv * L_cv + cfg.loss.lambda_rel * L_rel
            
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

            running += loss.item() * B
            
            log_dict = {
                'epoch': epoch,
                'step': global_step,
                'lr': lr_scheduler.get_last_lr()[0],
                'train/loss': float(loss.item()),
                'train/loss_dyn': float(L_dyn.item()), 
                'train/loss_event': float(L_evt.item()),
                'train/loss_crossview': float(L_cv if isinstance(L_cv, float) else L_cv.item()),
                'train/loss_rel': float(L_rel if isinstance(L_rel, float) else L_rel.item()),
                'train/S_mean': float(s.mean().item()),
                'train/dR_mean': float(dR.mean().item()),   
            }
            run.log(log_dict)
            json_log(log_dict)
            
            if max_train_steps is not None and train_step >= max_train_steps:
                break
            
            lr_scheduler.step()
            train_step += 1
            global_step += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.3f}'})
        
        if epoch % cfg.train.val_every == 0:
            enc.eval()
            dyn.eval()
            evt.eval()
            val_step = 0
            with torch.no_grad():
                val_loss, nval = 0.0, 0.0
                for batch in tqdm(val_loader, desc=f'Validation at epoch {epoch}', leave=False):
                    imgs = []
                    for v in views:
                        img = batch[v]
                        imgs.append(img)
                    imgs = torch.stack(imgs, dim=1)         # (B, V, T, C, H, W)
                    imgs = imgs.to(device)

                    z_views = [enc(imgs[:, v]) for v in range(len(views))]      
                    z = torch.stack(z_views, dim=1)
                    z_gt = z[:, :, -1].mean(dim=1)
                    z_in = z[:, :, :-1].mean(dim=1)     # (B, T-1, S, D)
                    z_pred = dyn(z_in)
                    L_dyn = (z_pred - z_gt).pow(2).mean()
                    val_loss += L_dyn.item() * imgs.size(0)
                    nval += imgs.size(0)
                    val_step += 1
                    if max_val_steps is not None and val_step >= max_val_steps:
                        break
                val_loss /= max(nval, 1)

            run.log({'validation/loss_dyn': val_loss, 'epoch': epoch, 'step': global_step})
            json_log({'epoch': epoch, 'step': global_step, 'valdation/loss_dyn': val_loss})
            
            if val_loss < best:
                state = {
                    'epoch': epoch,
                    'encoder': enc.state_dict(),
                    'dynamics': dyn.state_dict(),
                    'event_detector': evt.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': lr_scheduler.state_dict(),
                    'config': dict(cfg),
                }
                save_ckpt_state(state=state, path=os.path.join(output_dir, 'ckpt', f'{epoch:03d}_val_loss={val_loss:.3f}.pt'))
                best = val_loss
            
            
        if (epoch % cfg.train.ckpt_every == 0) and cfg.train.save_last_ckpt:
            state = {
                'epoch': epoch,
                'encoder': enc.state_dict(),
                'dynamics': dyn.state_dict(),
                'event_detector': evt.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': lr_scheduler.state_dict(),
                'config': dict(cfg),
            }
            save_ckpt_state(state=state, path=os.path.join(output_dir, 'ckpt', f'latest.pt'))
        
        if max_train_epoch is not None and epoch >= max_train_epoch:
            break
        
    run.finish()
    print(f'Train finished at {datetime.now().isoformat()}. Results are save at {output_dir}.')
            

if __name__=='__main__':
    main()