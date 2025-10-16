import torch

def pixel_norm(x: torch.Tensor, dim: int=-1, eps: float=1e-4) -> torch.Tensor:
    """Normalize input such that it has length sqrt(D)."""
    return x / torch.sqrt(torch.mean(x**2, dim=dim, keepdim=True) + eps)