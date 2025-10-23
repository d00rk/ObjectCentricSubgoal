import torch

def pixel_norm(x: torch.Tensor, dim: int=-1, eps: float=1e-4) -> torch.Tensor:
    """Normalize input such that it has length sqrt(D)."""
    return x / torch.sqrt(torch.mean(x**2, dim=dim, keepdim=True) + eps)


def to01(x: torch.Tensor) -> torch.Tensor:
    """
    [0, 255] Image or [-1, 1] Image -> [0, 1] Image
    """
    x = x.float()
    mx, mn = float(x.max()), float(x.min())
    if mx > 1.5:
        x = x / 255.0
    elif mn < -0.2:
        x = (x + 1.0) / 2.0
    return x.clamp(0, 1)