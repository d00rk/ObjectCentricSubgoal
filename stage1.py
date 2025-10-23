import os, sys
from datetime import datetime
import hydra
from omegaconf import OmegaConf, DictConfig
import tqdm
import math
import wandb
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from torchvision.utils import save_image

from slot.utils.common import set_seed
from slot.utils.neural_networks import get_scheduler
from slot.utils.pytorch_uilts import optimizer_to, get_learnable_params
from slot.utils.logging import JsonLogger
from slot.utils.checkpoint import TopKCheckpointManager, save_checkpoint, load_checkpoint
from slot.utils.normalization import to01
from slot.utils.visualization import overlay_heatmap_on_image


@hydra.main(version_base=None, config_path='./slot/config', config_name='stage1.yaml')
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
    base_params = [p for p in learnable_params if p is not policy.logit_scale]
    logit_params = [policy.logit_scale]
    assert all(id(p) != id(logit_params[0]) for p in base_params)

    param_group = [
        {"params": base_params, "" "weight_decay": cfg.optimizer.weight_decay, },
        {"params": logit_params, "weight_decay": 0.0},
    ]
    
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=param_group)

    print(f"Total Learnable Parameters {sum(p.numel() for p in learnable_params)/1e6:.3f}M")
    
    lr_scheduler = get_scheduler(cfg.lr_scheduler.name, optimizer=optimizer, num_warmup_steps=cfg.lr_scheduler.num_warmup_steps,
                                 num_training_steps=(len(train_dataloader)*cfg.training.num_epochs), last_epoch=-1)
    
    if cfg.training.resume:
        ckpt_path = cfg.training.resume_ckpt
        if ckpt_path.is_file():
            print(f"Resuming from checkpoint: {ckpt_path}")
            cfg = load_checkpoint(ckpt_path, model=policy, optimizer=optimizer, lr_scheduler=lr_scheduler)
            learnable_params = list(get_learnable_params(policy))
            
    policy.to(device)
    optimizer_to(optimizer, device)
    
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
    for epoch in range(int(cfg.training.num_epochs)):
        train_pbar = tqdm.tqdm(train_dataloader, desc=f'Training', leave=False, mininterval=cfg.training.tqdm_interval_sec)
        policy.train()
        total_loss, total_recon, total_con = list(), list(), list()
        step_log = dict()
        for batch_idx, batch in enumerate(train_pbar):
            imgs = batch['current_image'].to(device, non_blocking=True)
            instructions = batch['instruction']
            
            output = policy(imgs, instructions)

            # Feature reconstruction loss
            gt_feature = output['visual_tokens']
            B, P, _ = gt_feature.shape
            patch = int(math.sqrt(P))
            assert patch*patch == P
            gt_feature = gt_feature.view(B, patch, patch, -1)
            gt_feature = F.normalize(gt_feature.float(), dim=-1)
            reconstructed_feature = output['reconstruction']
            reconstructed_feature = F.normalize(reconstructed_feature.float(), dim=-1)
            loss_recon = F.mse_loss(gt_feature, reconstructed_feature)
            
            # Contrastive loss
            attn = output['attn'].float()                            # (B, num_slots, num_patches)
            image_tokens = output['mapped_visual_tokens'].float()    # (B, num_patches, feature_dim)
            slot_embedding = torch.einsum('bkp,bpd->bkd', attn, image_tokens)
            slot_embedding = F.normalize(slot_embedding.float(), dim=-1)    # (B, S, feature_dim)
            text_features = F.normalize(output['text_features'].float(), dim=-1)
            
            sim_bkb = torch.einsum('bkd,jd->bkj', slot_embedding, text_features)
            logit_scale_exp = policy.logit_scale.exp().clamp(max=100.0)
            logits_i2t = sim_bkb.max(dim=1).values * logit_scale_exp
            logits_t2i = logits_i2t.t()
            
            targets = torch.arange(logits_i2t.size(0), device=device)
            loss_i2t = F.cross_entropy(logits_i2t, targets)
            loss_t2i = F.cross_entropy(logits_t2i, targets)
            loss_contrastive = 0.5 * (loss_i2t + loss_t2i)
            
            loss_total = loss_recon + cfg.loss.lambda_contrastive * loss_contrastive
            
            optimizer.zero_grad(set_to_none=True)
            loss_total.backward()
            clip_grad_norm_(learnable_params, 5.0)
            optimizer.step()
            lr_scheduler.step()
            
            raw_loss_total = loss_total.item()
            raw_loss_recon = loss_recon.item()
            raw_loss_cont = loss_contrastive.item()
            train_pbar.set_postfix(loss=raw_loss_total, refresh=False)
            total_loss.append(raw_loss_total)
            total_recon.append(raw_loss_recon)
            total_con.append(raw_loss_cont)
            
            step_log = {
                'train/loss': raw_loss_total,
                'train/loss_recon': raw_loss_recon,
                'train/loss_contrastive': raw_loss_cont,
                'debug/logit_scale_exp': logit_scale_exp.item(),
                'global_step': global_step,
                'epoch': epoch,
                'lr': lr_scheduler.get_last_lr()[0]
            }
            
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
        train_recon_loss = np.mean(total_recon)
        train_cont_loss = np.mean(total_con)
        step_log['train/loss'] = train_loss
        step_log['train/loss_recon'] = train_recon_loss
        step_log['train/loss_contrastive'] = train_cont_loss
        
        if epoch % cfg.training.val_every == 0:
            policy.eval()
            with torch.no_grad():
                total_val_loss, total_val_recon, total_val_cont = list(), list(), list()
                val_pbar = tqdm.tqdm(val_dataloader, desc=f'Validation', leave=False, mininterval=cfg.training.tqdm_interval_sec)
                for batch_idx, batch in enumerate(val_pbar):
                    imgs = batch['current_image'].float().to(device)
                    instructions = batch['instruction']
                    B, C, H, W = imgs.shape
                    
                    output = policy(imgs, instructions)

                    # Feature reconstruction loss
                    gt_feature = output['visual_tokens']
                    _, P, _ = gt_feature.shape
                    patch = int(math.sqrt(P))
                    assert patch*patch == P
                    gt_feature = gt_feature.view(B, patch, patch, -1)
                    gt_feature = F.normalize(gt_feature, dim=-1)
                    reconstructed_feature = output['reconstruction']
                    reconstructed_feature = F.normalize(reconstructed_feature, dim=-1)
                    loss_recon = F.mse_loss(gt_feature, reconstructed_feature)

                    # Contrastive loss
                    attn = output['attn'].float()                            # (B, num_slots, num_patches)
                    image_tokens = output['mapped_visual_tokens'].float()    # (B, num_patches, feature_dim)
                    slot_embedding = torch.einsum('bkp,bpd->bkd', attn, image_tokens)
                    slot_embedding = F.normalize(slot_embedding, dim=-1)     # (B, )
                    
                    text_features = F.normalize(output['text_features'].float(), dim=-1)
                    
                    sim_bkb = torch.einsum('bkd,jd->bkj', slot_embedding, text_features)
                    logit_scale_exp = policy.logit_scale.exp().clamp(max=100.0)
                    logits_i2t = sim_bkb.max(dim=1).values * logit_scale_exp
                    logits_t2i = logits_i2t.t()
                    
                    targets = torch.arange(logits_i2t.size(0), device=device)
                    loss_i2t = F.cross_entropy(logits_i2t, targets)
                    loss_t2i = F.cross_entropy(logits_t2i, targets)
                    loss_contrastive = 0.5 * (loss_i2t + loss_t2i)
                    
                    loss_total = loss_recon + cfg.loss.lambda_contrastive * loss_contrastive
                    
                    total_val_loss.append(loss_total.item())
                    total_val_recon.append(loss_recon.item())
                    total_val_cont.append(loss_contrastive.item())
                    
                    # visualization for debugging
                    if batch_idx < 4:
                        cos_map = F.cosine_similarity(reconstructed_feature, gt_feature, dim=-1)        # (B, patch, patch)
                        cos_img = (cos_map.clamp(-1, 1)*0.5 + 0.5).unsqueeze(1)
                        cos_up = F.interpolate(cos_img, size=(H, W), mode='bilinear', align_corners=True)
                        save_image(cos_up[:8].cpu(), os.path.join(vis_dir, f'val_epoch{epoch:02d}_batch={batch_idx:02d}_heatmap.png'), nrow=4)
                        
                        sim = torch.einsum('sd,td->st', slot_embedding[0], text_features[0].unsqueeze(0)).squeeze(-1)
                        topv, topi = torch.topk(sim, k=min(6, sim.numel()))
                        step_log.update({
                            f'validation/top_sim_slot_{i}': v.item() for i, v in zip(topi.tolist(), topv.tolist())
                        })
                        
                    attn = output['attn']       # (B, n_slots, num_patches)
                    _, S, _ = attn.shape
                    attn = attn.view(B, S, patch, patch)
                    heat_up = F.interpolate(attn, size=(H, W), mode='bilinear', align_corners=True)
                    x_gt = to01(imgs)
                    for i in range(4):
                        ov = overlay_heatmap_on_image(x_gt[i].cpu(), heat_up[i].cpu(), alpha=0.35)
                        save_image(ov, os.path.join(vis_dir, f"val_epoch{epoch:02d}_index{i:02d}_slot_attention_overlay.png"), nrow=4)
                    
                    if cfg.training.max_val_steps is not None and batch_idx >= cfg.training.max_val_steps-1:
                        break
                
                val_loss = np.mean(total_val_loss)
                val_recon_loss = np.mean(total_val_recon)
                val_cont_loss = np.mean(total_val_cont)
                step_log['validation/loss'] = val_loss
                step_log['validation/loss_recon'] = val_recon_loss
                step_log['validation/loss_contrastive'] = val_cont_loss
        
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