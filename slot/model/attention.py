from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from slot.utils.neural_networks import build_mlp


class SelfAttnBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        dropout: float=0.0, 
        mlp_ratio: float=4.0
    ):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.mlp = build_mlp(input_dim=dim, output_dim=dim, features=[int(dim*2)], activation_fn='gelu', initial_layer_norm=True, residual=True, dropout=dropout)
        
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        # x: (B, S, D)
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + self.drop(h)
        x = self.mlp(x)
        return x


class CrossAttnBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        dropout: float=0.0,
        mlp_ratio: float=4.0,
    ):
        super().__init__()
        self.ln_q = nn.LayerNorm(dim)
        self.ln_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.mlp = build_mlp(input_dim=dim, output_dim=dim, features=[int(dim*2)], activation_fn='gelu', initial_layer_norm=True, residual=True, dropout=dropout)
    
    def forward(self, 
                x: torch.Tensor, 
                lang: torch.Tensor, 
                lang_padding_mask: Optional[torch.Tensor]=None, 
                return_attn: bool=False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        q = self.ln_q(x)
        kv = self.ln_kv(lang)
        h, attn_w = self.attn(q, kv, kv, key_padding_mask=lang_padding_mask, need_weights=return_attn, average_attn_weights=False)
        x = x + self.drop(h)
        x = self.mlp(x)
        return x, (attn_w if return_attn else None)


class SlotPixeleAttention(nn.Module):
    def __init__(self, c: int, d: int, d_attn: int=128, tau: float=0.25, norm_over: str='slot'):
        super().__init__()
        self.key = nn.Conv2d(c, d_attn, 1)
        self.query = nn.Linear(d, d_attn)
        self.tau = tau
        assert norm_over in ("slot", "spatial")
        self.norm_over = norm_over
    
    def forward(self, feat, slots):
        B, C, H, W = feat.shape
        K = slots.shape[1]
        
        key = self.key(feat)
        key = key.view(B, -1, H*W)
        key = F.normalize(key, dim=1)
        
        query = self.query(slots)
        query = F.normalize(query, dim=-1)
        
        logits = torch.einsum('bkd,bdl->bkl', query, key) / max(self.tau, 1e-6)
        
        if self.norm_over == 'slot':
            attn = F.softmax(logits, dim=1)
        else:
            attn = F.softmax(logits, dim=2)
            
        heat = attn.view(B, K, H, W)
        return heat