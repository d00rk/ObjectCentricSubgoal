from typing import Any, Dict, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from slot.model.attention import SelfAttnBlock, CrossAttnBlock

class SlotTransformer(nn.Module):
    def __init__(
        self,
        dim: int,
        language_dim: int=512,
        n_layers: int=3,
        n_heads: int=3,
        mlp_ratio: float=4.0,
        dropout: float=0.0,
        residual_to_input: bool=True,
        return_attn: bool=False,
    ):
        super().__init__()
        
        self.return_attn = return_attn
        self.residual_to_input = residual_to_input
        
        self.proj_lang = nn.Linear(language_dim, dim)   # project language embedding to model space.
        
        self.sa_blocks = nn.ModuleList([
            SelfAttnBlock(dim, n_heads, dropout, mlp_ratio) for _ in range(n_layers)
        ])
        self.ca_blocks = nn.ModuleList([
            CrossAttnBlock(dim, n_heads, dropout, mlp_ratio) for _ in range(n_layers)
        ])
        
        self.pred_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim)
        )
    
    @staticmethod
    def _build_key_padding_mask(lang_mask: Optional[torch.Tensor]) ->Optional[torch.Tensor]:
        """
        Build key_padding mask(True=pad) for multi-head attention.
        """
        if lang_mask is None:
            return None
        
        if lang_mask.dtype != torch.bool:
            kpm = (lang_mask == 0)
        else:
            kpm = ~lang_mask
        return kpm

    def _prepare_language(self, lang_embeddings: torch.Tensor, lang_masks: Optional[torch.Tensor]):
        if lang_embeddings.dim() == 2:
            lang_embeddings = lang_embeddings.unsqueeze(1)
            if lang_masks is None:
                lang_masks = torch.ones((lang_embeddings.size(0), 1), dtype=torch.long, device=lang_embeddings.device)
        
        L = self.proj_lang(lang_embeddings)
        kpm = self._build_key_padding_mask(lang_masks)
        return L, kpm
    
    def forward(
        self,
        slots: torch.Tensor,            # (B, S, D)
        lang_embeddings: torch.Tensor,  # (B, 512)
        lang_masks: Optional[torch.Tensor]=None,
    ) -> Dict[str, Any]:
        assert slots.dim() == 3, f"Slots must be (B, S, D), not {slots.shape}"
        
        L, lang_kpm = self._prepare_language(lang_embeddings, lang_masks)
        
        x = slots
        cross_attn_weights: List[torch.Tensor] = []
        for self_blk, cross_blk in zip(self.sa_blocks, self.ca_blocks):
            x = self_blk(x)
            x, attn_w = cross_blk(x, L, lang_kpm, return_attn=self.return_attn)
            if self.return_attn and attn_w is not None:
                cross_attn_weights.append(attn_w)
            
        delta = self.pred_head(x)
        y = slots + delta if self.residual_to_input else x + delta
        
        out = {'slots': y}
        if self.return_attn:
            out['cross_attn'] = cross_attn_weights
        return out