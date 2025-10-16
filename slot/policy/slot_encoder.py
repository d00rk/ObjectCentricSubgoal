from typing import Optional, Tuple, Dict
from dataclasses import dataclass
import math
import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from slot.utils.neural_networks import build_grid_of_positions
from slot.model.decoder import get_slotattention_decoder_backbone, SlotAttentionDecoder
from slot.model.slot_attention import SlotAttention, ControllableSlotAttentionGrouping


class SlotEncoder(nn.Module):
    """
    Slot Encoder Model.
    - Visual: R3M/VIP/LIV encoder (frozen) -> Mapping g -> CLIP image space
    - Text: CLIP text encoder (frozen)
    - Slot Attention: insturction-conditioned object slots
    - Decoder: CLIP feature reconstruction from slots
    """
    def __init__(
        self,
        image_feature_extractor: str='r3m',     # 'r3m', 'vip', 'liv', 'clip'
        image_feature_extractor_backbone_name: str='resnet50',
        freeze_image_feature_extractor: bool=True,
        text_feature_extractor: str='clip',
        text_feature_extractor_backbone_name: str='RN50',
        freeze_text_feature_extractor: bool=True,
        mapping: str='mlp',     # 'mlp', 'identity'
        slot_dim: int=256,
        n_slots: int=8,
        n_lang_slots: int=4,
        n_heads: int=1,
        n_iters: int=3,
    ):
        super().__init__()
        image_feature_extractor = image_feature_extractor.lower()
        assert image_feature_extractor in ['r3m', 'vip', 'liv', 'clip'], f"{image_feature_extractor} not supported for visual encoding."
        
        text_feature_extractor = text_feature_extractor.lower()
        assert text_feature_extractor in ['clip'], f"{text_feature_extractor} not supported for text encoding."
        
        mapping = mapping.lower()
        assert mapping in ['identity', 'mlp'], f"{mapping} not supported for mapping."
        
        if image_feature_extractor == 'r3m':
            from slot.model.feature_extractor.r3m_extractor import R3MImageFeatureExtractor
            self.visaul_encoder = R3MImageFeatureExtractor(model_type=image_feature_extractor_backbone_name, freeze=freeze_image_feature_extractor)
        elif image_feature_extractor == 'vip':
            from slot.model.feature_extractor.vip_extractor import VIPImageFeatureExtractor
            self.visaul_encoder = VIPImageFeatureExtractor(model_type=image_feature_extractor_backbone_name, freeze=freeze_image_feature_extractor)
        elif image_feature_extractor == 'liv':
            from slot.model.feature_extractor.liv_extractor import LIVImageFeatureExtractor
            self.visaul_encoder = LIVImageFeatureExtractor(model_type=image_feature_extractor_backbone_name, freeze=freeze_image_feature_extractor)
        elif image_feature_extractor == 'clip':
            from slot.model.feature_extractor.clip_extractor import CLIPImageFeatureExtractor
            self.visaul_encoder = CLIPImageFeatureExtractor(model_type=image_feature_extractor_backbone_name, freeze=freeze_image_feature_extractor)
        else:
            self.visaul_encoder = None
        
        
        if text_feature_extractor == 'clip':
            from slot.model.feature_extractor.clip_extractor import CLIPTextModel
            self.text_encoder = CLIPTextModel(model_type=text_feature_extractor_backbone_name, freeze=freeze_text_feature_extractor)
        else:
            self.text_encoder = None
            
        if mapping == 'mlp':
            from slot.model.mapping import MLPMapping
            self.mapping = MLPMapping(dim=self.visaul_encoder.out_dim)
        elif mapping == 'identitiy':
            from slot.model.mapping import IdentityMapping
            self.mapping = IdentityMapping()
        else:
            self.mapping = None
            
        self.text_projection = nn.Sequential(
            nn.Linear(self.text_encoder.out_dim, slot_dim),
            nn.LayerNorm(slot_dim),
            nn.GELU(),
            nn.Linear(slot_dim, slot_dim)
        )
        
        self.slot_attention = SlotAttention(
            dim=slot_dim,
            feature_dim=self.visaul_encoder.out_dim,
            n_heads=n_heads,
            n_iters=n_iters,
            lang_conditioning=True,
            use_query_init=True,
            num_slots=n_slots,
            num_lang_slots=n_lang_slots,
            text_dim=slot_dim,
        )
        
        decoder_backbone = get_slotattention_decoder_backbone(object_dim=slot_dim, output_dim=4)
        self.decoder = SlotAttentionDecoder(decoder=decoder_backbone, final_activation='gelu')
    
    def forward(self, images: torch.Tensor, instructions: str) -> Dict[str, torch.Tensor]:
        B, C, H, W = images.shape
        device = images.device
        dtype = images.dtype
        
        visual_features = self.visaul_encoder(images)                       # (B, D_vis)
        mapped_visual_features = self.mapping(visual_features)              # (B, D_m)
        mapped_visual_features = F.normalize(mapped_visual_features, dim=-1)
        
        text_tokens = torch.stack([clip.tokenize(instr) for instr in instructions], dim=0)
        text_features = torch.stack([self.text_encoder(instr).squeeze(0) for instr in instructions]).to(device)             # (B, D_str)
        text_projected_features = self.text_projection(text_features)
        text_projected_features = F.normalize(text_projected_features, dim=-1)
        
        slots, attn, attn_list = self.slot_attention(
            inputs=mapped_visual_features.unsqueeze(1),
            text_embeddings=text_projected_features.unsqueeze(1)
        )
        
        recon = self.decoder(slots.mean(1))
        recon = F.normalize(recon, dim=-1)

        return {
            'slots': slots,
            'attn': attn,
            'recon': recon,
            'visual_features': visual_features,
            'mapped_visual_features': mapped_visual_features,
            'text_features': text_features
        }
        
        
        
if __name__ == '__main__':
    import numpy as np
    model = SlotEncoder()
    instructions = ['a photo of bag', 'a photo of cat', 'a photo of dog', 'put the scissors in the pencil case']
    tokens = torch.stack([clip.tokenize(instr).squeeze(0) for instr in instructions], dim=0)
    print(tokens.shape)