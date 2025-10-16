import torch
import torch.nn as nn

class MLPMapping(nn.Module):
    def __init__(
        self,
        dim: int,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, 2*dim),
            nn.LayerNorm(2*dim),
            nn.GELU(),
            nn.Linear(2*dim, dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

class IdentityMapping(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x