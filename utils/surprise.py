import torch

def pairwise_center_dist(x):
    # x: (B, S, D) -> (B, S, S)
    B, S, D = x.shape
    diff = x.unsqueeze(2) - x.unsqueeze(1)
    return diff.pow(2).sum(-1).sqrt()

def relational_change(z_cur, z_next):
    d_cur = pairwise_center_dist(z_cur)
    d_next = pairwise_center_dist(z_next)
    return (d_next - d_cur).pow(2).sum(dim=(1, 2)).sqrt()

def predictive_surprise(z_pred, z):
    return (z_pred - z).pow(2).sum(dim=(1,2))

def multiview_predictive_surprise(z_pred, z, reduce='mean'):
    s = (z_pred - z).pow(2).sum(dim=(2,3))
    if reduce == 'mean':
        return s.mean(dim=1)
    elif reduce == 'max':
        return s.max(dim=1).values
    else:
        return s

def multiview_relational_change(z_cur, z_next, reduce='mean'):
    B, V, S, D = z_cur.shape
    vals = []
    for v in range(V):
        vals.append(relational_change(z_cur[:, v], z_next[:, v]))
    dR = torch.stack(vals, dim=1)   # (B, V)
    if reduce == 'mean':
        return dR.mean(dim=1)
    elif reduce == 'max':
        return dR.max(dim=1).values
    else:
        return dR
    
def soft_labels(S, dR, ema_mean, ema_std, temp=5.0, rel_w=0.5):
    x = (S + rel_w * dR - ema_mean) / (ema_std + 1e-6)
    return torch.sigmoid(temp * x)