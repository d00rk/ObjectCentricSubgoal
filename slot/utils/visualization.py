import matplotlib.cm as cm
import torch


def overlay_heatmap_on_image(image: torch.Tensor, heat: torch.Tensor, alpha: float=0.5):
    """
    Image (3, H, W) or (1, H, W), float in [0, 1]
    Heat: (K, H, W), float
    return: (K, 3, H, W)
    """
    assert image.dim() == 3 and heat.dim() == 3, f"Image {image.shape}, Heat {heat.shape}"
    if image.size(0) == 1:
        image = image.repeat(3, 1, 1)
    image = image.to(dtype=torch.float32).clamp(0, 1)
    
    K, H, W = heat.shape
    over = []
    
    for k in range(K):
        h = heat[k]
        h = (h - h.min()) / (h.max() - h.min() + 1e-8)
        colored = cm.get_cmap('jet')(h.cpu().numpy())[..., :3]
        colored = torch.from_numpy(colored).permute(2, 0, 1).to(image.device).float()
        out = ((1-alpha)*image[0] + alpha*colored).clamp(0, 1)
        over.append(out)
    return torch.stack(over, dim=0)