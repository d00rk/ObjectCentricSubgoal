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

from slot.dataset.dataloader import get_libero_image_dataloader, get_libero_subgoal_image_dataloader
from slot.utils.common import set_seed
from slot.utils.neural_networks import get_scheduler
from slot.utils.pytorch_uilts import optimizer_to, get_learnable_params
from slot.utils.logging import JsonLogger
from slot.utils.checkpoint import TopKCheckpointManager, save_checkpoint, load_checkpoint


@hydra.main(version_base=None, config_path='./slot/config', config_name='stage1.yaml')
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    
    seed = cfg.training.seed
    set_seed(seed)
    
    stage = int(cfg.stage)
    device = torch.device(cfg.training.device)
    output_dir = str(cfg.logging.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, 'visualization')
    os.makedirs(vis_dir, exist_ok=True)
    log_path = os.path.join(output_dir, 'logs.json')
    
    if stage == 1:
        train_dataloader, val_dataloader = get_libero_image_dataloader(**cfg.dataset)
    elif stage == 2:
        train_dataloader, val_dataloader = get_libero_subgoal_image_dataloader(**cfg.dataset)
    
    policy = hydra.utils.instantiate(cfg.model)
    learnable_params = list(get_learnable_params(policy))
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=learnable_params)

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
        cfg.training.max_train_steps = 3
        cfg.training.max_val_steps = 3
        cfg.training.val_every = 1
        cfg.training.checkpoint_every = 1
    
    json_logger = JsonLogger(log_path)
    json_logger.start()
    global_step = 0
    
    start_time = datetime.now()
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
            reconstructed_feature = output['reconstruction']
            loss_recon = F.mse_loss(gt_feature, reconstructed_feature)
            
            # Contrastive loss
            attn = output['attn'].float()                            # (B, num_slots, num_patches)
            image_tokens = output['mapped_visual_tokens'].float()    # (B, num_patches, feature_dim)
            slot_embedding = torch.einsum('bkp,bpd->bkd', attn, image_tokens)
            slot_embedding = F.normalize(slot_embedding, dim=-1)    # (B, S, feature_dim)
            text_features = F.normalize(output['text_features'].float(), dim=-1)
            
            sim_bkb = torch.einsum('bkd,jd->bkj', slot_embedding, text_features)
            logits_i2t = sim_bkb.max(dim=1).values
            logits_i2t = logits_i2t / cfg.loss.tau
            logits_t2i = logits_i2t.t()
            
            targets = torch.arange(logits_i2t.size(0), device=device)
            loss_i2t = F.cross_entropy(logits_i2t, targets)
            loss_t2i = F.cross_entropy(logits_t2i, targets)
            loss_contrastive = 0.5 * (loss_i2t + loss_t2i)
            
            loss_total = loss_recon + cfg.loss.lambda_contrastive * loss_contrastive
            
            optimizer.zero_grad(set_to_none=True)
            loss_total.backward()
            clip_grad_norm_(learnable_params, 1.0)
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
                    imgs = batch['current_image'].to(device)
                    instructions = batch['instruction']
                    
                    output = policy(imgs, instructions)

                    # Feature reconstruction loss
                    gt_feature = output['visual_tokens']
                    B, P, _ = gt_feature.shape
                    patch = int(math.sqrt(P))
                    assert patch*patch == P
                    gt_feature = gt_feature.view(B, patch, patch, -1)
                    reconstructed_feature = output['reconstruction']
                    loss_recon = F.mse_loss(gt_feature, reconstructed_feature)
                    
                    cos_map = F.cosine_similarity(reconstructed_feature, F.normalize(gt_feature, dim=-1), dim=-1)
                    cos_img = (cos_map.clamp(-1, 1)*0.5 + 0.5)
                    cos_img = cos_img.unsqueeze(1).cpu()
                    
                    save_image(cos_img[:8, ...], os.path.join(vis_dir, f'heatmap_epoch={epoch:03d}_batch={batch_idx:03d}.png'), nrow=4)
                    
                    # Contrastive loss
                    attn = output['attn'].float()                            # (B, num_slots, num_patches)
                    image_tokens = output['mapped_visual_tokens'].float()    # (B, num_patches, feature_dim)
                    slot_embedding = torch.einsum('bkp,bpd->bkd', attn, image_tokens)
                    slot_embedding = F.normalize(slot_embedding, dim=-1)     # (B, )
                    
                    text_features = F.normalize(output['text_features'].float(), dim=-1)
                    
                    sim_bkb = torch.einsum('bkd,jd->bkj', slot_embedding, text_features)
                    logits_i2t = sim_bkb.max(dim=1).values / cfg.loss.tau
                    logits_t2i = logits_i2t.t()
                    
                    targets = torch.arange(logits_i2t.size(0), device=device)
                    loss_i2t = F.cross_entropy(logits_i2t, targets)
                    loss_t2i = F.cross_entropy(logits_t2i, targets)
                    loss_contrastive = 0.5 * (loss_i2t + loss_t2i)
                    
                    loss_total = loss_recon + cfg.loss.lambda_contrastive * loss_contrastive
                    
                    total_val_loss.append(loss_total.item())
                    total_val_recon.append(loss_recon.item())
                    total_val_cont.append(loss_contrastive.item())
                
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
    print(f"End training at {end_time:%Y-%m-%d %H:%M%S}")
    print(f"Saved logs at {output_dir}")

if __name__=='__main__':
    main()