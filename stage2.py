import os, sys
from datetime import datetime
import hydra
from omegaconf import OmegaConf, DictConfig
import tqdm
import math
import wandb
import lpips
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from torchvision.utils import save_image

from slot.model.attention import SlotPixeleAttention
from slot.utils.common import set_seed
from slot.utils.neural_networks import get_scheduler, align_targets, masked_huber, psnr
from slot.utils.pytorch_uilts import optimizer_to, get_learnable_params
from slot.utils.logging import JsonLogger
from slot.utils.checkpoint import TopKCheckpointManager, save_checkpoint, load_checkpoint
from slot.utils.visualization import overlay_heatmap_on_image
from slot.utils.normalization import to01


@hydra.main(version_base=None, config_path='./slot/config', config_name='stage2.yaml')
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    
    seed = cfg.training.seed
    set_seed(seed)

    device = torch.device(cfg.training.device)
    output_dir = str(cfg.logging.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, 'visualization')
    os.makedirs(vis_dir, exist_ok=True)
    log_path = os.path.join(output_dir, 'logs.json')
    
    train_dataloader, val_dataloader = hydra.utils.instantiate(cfg.dataset)
    
    policy = hydra.utils.instantiate(cfg.model)
    learnable_params = list(get_learnable_params(policy))
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=learnable_params)

    print(f"Total Learnable Parameters {sum(p.numel() for p in learnable_params)/1e6:.3f}M")
    
    lr_scheduler = get_scheduler(cfg.lr_scheduler.name, 
                                 optimizer=optimizer, 
                                 num_warmup_steps=cfg.lr_scheduler.num_warmup_steps,
                                 num_training_steps=(len(train_dataloader)*cfg.training.num_epochs), 
                                 last_epoch=-1)
    
    if cfg.training.resume:
        ckpt_path = cfg.training.resume_ckpt
        if os.path.isfile(ckpt_path):
            print(f"Resuming from checkpoint: {ckpt_path}")
            cfg = load_checkpoint(ckpt_path, model=policy, optimizer=optimizer, lr_scheduler=lr_scheduler)
            learnable_params = list(get_learnable_params(policy))
            
    policy.to(device)
    optimizer_to(optimizer, device)
    
    spa = None
    
    topk_manager = TopKCheckpointManager(save_dir=os.path.join(output_dir, 'checkpoints'), **cfg.checkpoint.topk)
    
    if cfg.logging.wandb:
        wandb_run = wandb.init(
            dir=output_dir,
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging.wandb_args
        )
        wandb.config.update(
            {
                'output_dir': output_dir
            }
        )
    
    if cfg.training.debug:
        cfg.training.num_epochs = 2
        cfg.training.max_train_steps = 20
        cfg.training.max_val_steps = 20
        cfg.training.val_every = 1
        cfg.training.checkpoint_every = 1
    
    json_logger = JsonLogger(log_path)
    json_logger.start()
    global_step = 0
    
    start_time = datetime.now()
    print(f"Output will be saved at {output_dir}")
    print(f"Start training for {cfg.training.num_epochs} epochs at {start_time:%Y-%m-%d %H:%M:%S}")
    lpi = lpips.LPIPS(net='vgg').to(device).eval() if policy.rgb_decoder is not None else None
    
    for epoch in range(int(cfg.training.num_epochs)):
        train_pbar = tqdm.tqdm(train_dataloader, desc=f'Training', leave=False, mininterval=cfg.training.tqdm_interval_sec)
        policy.train()
        
        total_loss, total_pred = list(), list()
        if not policy.freeze_decoder:
            total_recon = list()
        if policy.rgb_decoder is not None:
            total_rgb = list()
            
        step_log = dict()
        
        for batch_idx, batch in enumerate(train_pbar):
            current_imgs = batch['current_image'].float().to(device, non_blocking=True)     # (B, C, H, W)
            subgoal_imgs = batch['subgoal_image'].float().to(device, non_blocking=True)     # (B, C, H, W)
            instructions = batch['instruction']
            
            output = policy(current_imgs, instructions)
            
            cur_slots = output['current_slots'].float()         # (B, n_slots, slot_dim)
            pred_slots = output['pred_slots'].float()           # (B, n_slots, slot_dim)
            pred_image_tokens = output['pred_image_tokens']     # (B, patch, patch, feature_dim_original)
            
            goal_enc = policy.encode(subgoal_imgs, instructions)
            goal_slots = goal_enc['slots']                      # (B, n_slots, slot_dim)
            
            align = align_targets(
                basis_slots=cur_slots,
                target_slots=goal_slots,
                target_exists=None,
                sim_threshold=float(cfg.loss.sim_threshold)
            )
            goal_slots_aligned = align['target_slots_aligned']  # (B, n_slots, slot_dim)
            goal_mask = align['target_mask']                    # (B, n_slots)
            
            loss_pred = masked_huber(pred=pred_slots, 
                                     target=goal_slots_aligned, 
                                     mask=goal_mask,
                                     delta=float(cfg.loss.huber_delta))
            
            loss = loss_pred
            if not policy.freeze_decoder:
                goal_image_tokens = goal_enc['visual_tokens']       # (B, patch, patch, feature_dim_original)
                goal_image_tokens = F.normalize(goal_image_tokens.float(), dim=-1)
                pred_image_tokens = F.normalize(pred_image_tokens.float(), dim=-1)
                
                loss_recon = F.mse_loss(goal_image_tokens, pred_image_tokens)
                loss = loss + 0.5 * loss_recon
                raw_loss_recon = loss_recon.item()
                total_recon.append(raw_loss_recon)
            
            if policy.rgb_decoder is not None:
                decoded = policy.decode_rgb(cur_slots)
                
                decoded_rgb = decoded['reconstruction']     # (B, C, H, W)
                decoded_mask = decoded['masks']             # (B, n_slots, H, W)
                
                x_gt = to01(current_imgs)
                x_rec = to01(decoded_rgb)
                
                l1 = F.l1_loss(x_gt, x_rec)
                lp = lpi((x_rec*2-1), (x_gt*2-1)).mean()
                
                p = decoded_mask.clamp_min(1e-8)
                ent = (p * torch.log(p)).sum(dim=1).mean()
                loss_rgb = l1 + cfg.loss.rgb_lp * lp + cfg.loss.rgb_entropy_alpha * ent
                loss = loss + 0.5 * loss_rgb
                raw_loss_rgb = loss_rgb.item()
                total_rgb.append(raw_loss_rgb)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(learnable_params, 5.0)
            optimizer.step()
            lr_scheduler.step()
            
            raw_loss_total = loss.item()
            raw_loss_pred = loss_pred.item()
            train_pbar.set_postfix(loss=raw_loss_total, refresh=False)
            total_loss.append(raw_loss_total)
            total_pred.append(raw_loss_pred)
            
            step_log = {
                'train/loss': raw_loss_total,
                'train/loss_pred': raw_loss_pred,
                'global_step': global_step,
                'epoch': epoch,
                'lr': lr_scheduler.get_last_lr()[0]
            }
            if not policy.freeze_decoder:
                step_log['train/loss_recon'] = raw_loss_recon
            if policy.rgb_decoder is not None:
                step_log['train/loss_rgb_recon'] = raw_loss_rgb
            
            is_last_batch = (batch_idx == len(train_dataloader)-1)
            if not is_last_batch:
                if cfg.logging.wandb:
                    wandb_run.log(step_log, step=global_step)
                json_logger.log(step_log)
                global_step += 1
            
            if cfg.training.max_train_steps is not None and batch_idx >= cfg.training.max_train_steps-1:
                break
        
        # at the end of each epoch
        train_loss = np.mean(total_loss)
        train_pred_loss = np.mean(total_pred)
        step_log['train/loss'] = train_loss
        step_log['train/loss_pred'] = train_pred_loss
        
        if not policy.freeze_decoder:
            train_recon_loss = np.mean(total_recon)
            step_log['train/loss_recon'] = train_recon_loss
        if policy.rgb_decoder is not None:
            train_rgb_loss = np.mean(total_rgb)
            step_log['train/loss_rgb_recon'] = train_rgb_loss
            
        if epoch % cfg.training.val_every == 0:
            policy.eval()
            with torch.no_grad():
                total_val_loss, total_val_pred = list(), list()
                if not policy.freeze_decoder:
                    total_val_recon = list()
                if policy.rgb_decoder is not None:
                    total_val_rgb = list()
                    total_val_psnr = list()
                    
                val_pbar = tqdm.tqdm(val_dataloader, desc=f'Validation', leave=False, mininterval=cfg.training.tqdm_interval_sec)
                
                for batch_idx, batch in enumerate(val_pbar):
                    current_imgs = batch['current_image'].float().to(device, non_blocking=True)
                    subgoal_imgs = batch['subgoal_image'].float().to(device, non_blocking=True)
                    instructions = batch['instruction']
                    B, C, H, W = current_imgs.shape
                    
                    output = policy(current_imgs, instructions)
                    
                    cur_slots = output['current_slots'].float()
                    pred_slots = output['pred_slots'].float()
                    pred_image_tokens = output['pred_image_tokens']
                    
                    goal_enc = policy.encode(subgoal_imgs, instructions)
                    goal_slots = goal_enc['slots']
                    goal_image_tokens = goal_enc['visual_tokens']
                    
                    align = align_targets(
                        basis_slots=cur_slots,
                        target_slots=goal_slots,
                        target_exists=None,
                        sim_threshold=float(cfg.loss.sim_threshold)
                    )
                    goal_slots_aligned = align['target_slots_aligned']
                    goal_mask = align['target_mask']
                    
                    loss_pred = masked_huber(pred=pred_slots, target=goal_slots_aligned, mask=goal_mask,
                                             delta=float(cfg.loss.huber_delta))
                    
                    loss = loss_pred
                    if not policy.freeze_decoder:
                        goal_image_tokens = F.normalize(goal_image_tokens.float(), dim=-1)
                        pred_image_tokens = F.normalize(pred_image_tokens.float(), dim=-1)
                        
                        loss_recon = F.mse_loss(goal_image_tokens, pred_image_tokens)
                        loss = loss + 0.5 * loss_recon
                        raw_loss_recon = loss_recon.item()
                        total_val_recon.append(raw_loss_recon)
                        
                    if policy.rgb_decoder is not None:
                        decoded = policy.decode_rgb(cur_slots)
                        decoded_rgb = decoded['reconstruction']
                        decoded_mask = decoded['masks']
                        
                        x_gt = to01(current_imgs)
                        x_rec = to01(decoded_rgb)
                        
                        l1 = F.l1_loss(x_gt, x_rec)
                        lp = lpi((x_rec*2-1), (x_gt*2-1)).mean()
                        
                        p = decoded_mask.clamp_min(1e-8)
                        ent = (p * torch.log(p)).sum(dim=1).mean()
                        
                        loss_rgb = l1 + cfg.loss.rgb_lp * lp + cfg.loss.rgb_entropy_alpha * ent
                        loss = loss + 0.5 * loss_rgb
                        raw_loss_rgb = loss_rgb.item()
                        total_val_rgb.append(raw_loss_rgb)
                        
                        ps = psnr(x_gt, x_rec).mean()
                        total_val_psnr.append(float(ps.item()))
                        
                        if batch_idx == 0:
                            N = min(4, current_imgs.size(0))
                            gt = x_gt[:N].cpu()
                            recon = x_rec[:N].cpu()
                            
                            interleaved = torch.stack([gt, recon], dim=1).flatten(0, 1)
                            save_path = os.path.join(vis_dir, f'val_epoch{epoch:02d}_batch{batch_idx:02d}_recon.png')
                            save_image(interleaved, save_path, nrow=2)
                        
                    total_val_loss.append(loss.item())
                    total_val_pred.append(loss_pred.item())
                    
                    if spa is None:
                        _, _, _, D = goal_image_tokens.shape
                        spa = SlotPixeleAttention(c=D, d=goal_slots_aligned.shape[-1], d_attn=128).to(device)
                    
                    heat = spa(goal_image_tokens.permute(0, 3, 1, 2).contiguous(), goal_slots_aligned)      # (B, n_slots, patch, patch)
                    heat_up = F.interpolate(heat, size=(H, W), mode='bilinear', align_corners=True)
                    x_sub_gt = to01(subgoal_imgs)
                    ov = overlay_heatmap_on_image(x_sub_gt[0].cpu(), heat_up[0].cpu(), alpha=0.2)
                    save_image(ov, os.path.join(vis_dir, f"val_epoch{epoch:02d}_slot_head_overlay.png"), nrow=4)
                    
                    if cfg.training.max_val_steps is not None and batch_idx >= cfg.training.max_val_steps-1:
                        break
                
                val_loss = np.mean(total_val_loss)
                val_pred_loss = np.mean(total_val_pred)
                step_log['validation/loss'] = val_loss
                step_log['validation/loss_pred'] = val_pred_loss
                
                if not policy.freeze_decoder:
                    val_recon_loss = np.mean(total_val_recon)
                    step_log['validation/loss_recon'] = val_recon_loss
                if policy.rgb_decoder is not None:
                    val_rgb_loss = np.mean(total_val_rgb)
                    step_log['validation/loss_rgb_recon'] = val_rgb_loss
                    val_psnr = np.mean(total_val_psnr)
                    step_log['validation/psnr'] = val_psnr
        
        if epoch % cfg.training.checkpoint_every == 0:
            if cfg.checkpoint.save_last_ckpt:
                save_checkpoint(model=policy, optimizer=optimizer, lr_scheduler=lr_scheduler, cfg=cfg, path=os.path.join(output_dir, 'checkpoints', 'latest.ckpt'))
            
            metric_dict = dict()
            for k, v in step_log.items():
                new_key = k.replace('/', '_')
                metric_dict[new_key] = v
            
            topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
            
            if topk_ckpt_path is not None:
                save_checkpoint(model=policy, optimizer=optimizer, lr_scheduler=lr_scheduler, cfg=cfg, path=topk_ckpt_path)
            
            import gc; gc.collect()
            
        policy.train()
        if cfg.logging.wandb:
            wandb_run.log(step_log, step=global_step)
        json_logger.log(step_log)
        global_step += 1
    
    json_logger.stop()
    end_time = datetime.now()
    print(f"End training at {end_time:%Y-%m-%d %H:%M:%S}")
    print(f"Saved logs at {output_dir}")

if __name__=='__main__':
    main()