import torch
import torch.nn as nn
from slot.utils.neural_networks import build_transformer_encoder

class MLPMapping(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int=512,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class IdentityMapping(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class TransformerEncoderMapping(nn.Module):
    def __init__(
        self,
        dim: int,
        n_layers: int,
        n_heads: int,
    ):
        super().__init__()
        self._dim = dim
        self.linear = None
        self.encoder = build_transformer_encoder(dim, dim, n_layers=n_layers, n_heads=n_heads)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self._dim:
            if self.linear is None:
                self.linear = nn.Linear(x.shape[-1], self._dim)
                self.linear.to(x.device)
            x = self.linear(x)
            
        x = self.encoder(x)
        return x


if __name__=='__main__':
    mapping = TransformerEncoderMapping(dim=384, n_layers=3, n_heads=1)
    
    dummy = torch.zeros(1, 16, 2048)
    
    x = mapping(dummy)
    
    print(x.shape)
    