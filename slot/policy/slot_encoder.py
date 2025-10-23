from typing import Dict
import math
import clip
import torch
import torch.nn as nn
import torch.nn.functional as F

class SlotEncoder(nn.Module):
    """
    Slot Encoder Model.
    - Visual: R3M/VIP/LIV encoder (frozen) -> Mapping g -> query image space
    - Text: CLIP text encoder (frozen)
    - Slot Attention: insturction-conditioned object slots
    - Decoder: query feature reconstruction from slots
    """
    def __init__(
        self,
        visual_encoder: nn.Module,
        text_encoder: nn.Module,
        mapping: nn.Module,
        slot_attention: nn.Module,
        decoder: nn.Module,
    ):
        super().__init__()
        self.visual_encoder = visual_encoder
        self.text_encoder = text_encoder
        self.mapping = mapping
        self.slot_attention = slot_attention
        self.decoder = decoder
        
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1/0.07), dtype=torch.float32))
    
    def forward(self, images: torch.Tensor, instructions: str) -> Dict[str, torch.Tensor]:
        encoder_out = self.encode(images, instructions)
        slots = encoder_out['slots']
        recon = self.decode(slots)
        pred = recon['reconstruction']
        
        return {
            'slots': slots,
            'attn': encoder_out['attn'],
            'reconstruction': pred,
            'masks': recon['masks'],
            'masks_as_image': recon['masks_as_image'],
            'per_slot_patches': recon['per_slot_patches'],      # (B, num_slots, num_patches, feature_dim_original)
            'visual_tokens': encoder_out['visual_tokens'],
            'mapped_visual_tokens': encoder_out['mapped_visual_tokens'],
            'text_features': encoder_out['text_features'],
        }
    
    def encode(self, images: torch.Tensor, instructions: str):
        device = images.device
        dtype = images.dtype
        if dtype == torch.uint8:
            images = images.float().div_(255.0)
            dtype = images.dtype
        
        visual_features, visual_tokens = self.visual_encoder(images, return_tokens=True)    # tokens: (B, n_patches, channel)
        visual_tokens = visual_tokens.to(device=device, dtype=dtype)
        
        text_tokens = clip.tokenize(instructions, truncate=True).to(device, non_blocking=True)
        text_features = self.text_encoder(text_tokens)
        text_features = text_features.to(dtype=dtype)
        text_features = F.normalize(text_features, dim=-1)                      # (B, text_dim)
        
        mapped_visual_tokens = self.mapping(visual_tokens)                      # (B, n_patches, feature_dim)
        mapped_visual_tokens_n = F.normalize(mapped_visual_tokens, dim=-1)        # (B, n_patches, feature_dim)
        
        slots, attn, attn_list = self.slot_attention(
            inputs=mapped_visual_tokens_n,
            text_embeddings=text_features.unsqueeze(1)
        )
        
        return {
            'slots': slots,
            'attn': attn,
            'visual_tokens': visual_tokens,
            'mapped_visual_tokens': mapped_visual_tokens,
            'text_features': text_features,
        }
    
    def decode(self, slots: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = slots.shape[0]
        
        recon = self.decoder(slots)
        pred = recon['reconstruction']      # reconstruction: (B, num_patches, feature_dim_original), masks: (B, n_slots, num_patches)
        num_patches = pred.shape[1]
        patch = int(math.sqrt(num_patches))
        pred = pred.view(B, patch, patch, -1)       # (B, num_patches**1/2, num_patches**1/2, feature_dim_original)
        
        return {
            'reconstruction': pred,
            'masks': recon['masks'],
            'masks_as_image': recon['masks_as_image'],
            'per_slot_patches': recon['per_slot_patches'],      # (B, num_slots, num_patches, feature_dim_original)
        }