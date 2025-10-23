from typing import Dict, Optional, List, Callable, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR, LinearLR
from diffusers.optimization import Union, SchedulerType, Optional, TYPE_TO_SCHEDULER_FUNCTION


class RELUSquared(nn.Module):
    def __init__(self, inplace: bool=False):
        super().__init__()
        self.inplace = inplace
        
    def forward(self, x):
        return nn.functional.relu(x, inplace=self.inplace) ** 2


class ReGLU(nn.Module):
    """ ReLU + GLU activation. """
    @staticmethod
    def reglu(x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] % 2 == 0
        a, b  = x.chunk(2, dim=-1)
        return a * nn.functional.relu(b)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.reglu(x)


class GEGLU(nn.Module):
    """ GELU + GLU activation. """
    @staticmethod
    def geglu(x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] % 2 == 0 
        a, b = x.chunk(2, dim=-1)
        return a * nn.functional.gelu(b)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.geglu(x)


def get_activation_fn(name: str, inplace: bool=True, leaky_relu_slope: Optional[float]=None):
    if callable(name):
        return name
    
    name = name.lower()
    if name == 'relu':
        return nn.ReLU(inplace=inplace)
    elif name == 'relu_squared':
        return RELUSquared(inplace=inplace)
    elif name == 'leaky_relu':
        assert leaky_relu_slope is not None, f"Slope of leaky ReLU was not defined."
        return nn.LeakyReLU(leaky_relu_slope, inplace=inplace)
    elif name == 'gelu':
        return nn.GELU()
    elif name == 'glu':
        return nn.GLU()
    elif name == 'geglu':
        return GEGLU()
    elif name == 'reglu':
        return ReGLU()
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'identity':
        return nn.Identity()
    else:
        raise ValueError(f"Unknown activation function: {name}")


def CosineAnnealingWarmUpRestarts(optimizer, T_max, T_warmup=2000, start_factor=0.1, eta_min=1e-4):
    warmup_scheduler = LinearLR(optimizer, start_factor=start_factor, total_iters=T_warmup)
    annealing_scheduler = CosineAnnealingLR(optimizer, T_max=T_max-T_warmup, eta_min=eta_min)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, annealing_scheduler], milestones=[T_warmup])
    return scheduler


def get_scheduler(name, optimizer, num_warmup_steps, num_training_steps, **kwargs):
    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
    if name == SchedulerType.CONSTANT:
        return schedule_func(optimizer, **kwargs)
    
    if num_warmup_steps is None:
        raise ValueError(f"{name} requires 'num_warmup_steps', please provied that argument.")
    
    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, **kwargs)
    
    if num_training_steps is None:
        raise ValueError(f"{name} requires 'num_training_steps', please provide that argument.")
    
    return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, **kwargs)


def build_grid_of_positions(resolution, min=0.0):
    ranges = [torch.linspace(min, 1.0, steps=res) for res in resolution]
    grid = torch.meshgrid(*ranges, indexing='ij')
    grid = torch.stack(grid, dim=-1)
    grid = torch.reshape(grid, [resolution[0], resolution[1], -1])
    return grid


def build_transformer_encoder(input_dim: int, 
                              output_dim: int, 
                              n_layers: int, 
                              n_heads: int, 
                              hidden_dim: Optional[int]=None,
                              dropout: float=0.0,
                              activation_fn: Union[str, callable]='relu',
                              layer_norm_eps: float=1e-5,
                              use_output_transform: bool=True):
    if hidden_dim is None:
        hidden_dim = 4 * input_dim
        
    layers = []
    for _ in range(n_layers):
        layers.append(
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                activation=activation_fn,
                layer_norm_eps=layer_norm_eps,
                batch_first=True,
                norm_first=True
            )
        )
    
    if use_output_transform:
        layers.append(nn.LayerNorm(input_dim, eps=layer_norm_eps))
        output_transform = nn.Linear(input_dim, output_dim, bias=True)
        nn.init.xavier_uniform_(output_transform.weight)
        nn.init.zeros_(output_transform.bias)
        layers.append(output_transform)
    
    return nn.Sequential(*layers)


class Residual(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return inputs + self.module(inputs)


def build_mlp(
    input_dim: int,
    output_dim: int,
    features: List[int],
    activation_fn: Union[str, Callable] = "relu",
    final_activation_fn: Optional[Union[str, Callable]] = None,
    initial_layer_norm: bool = False,
    residual: bool = False,
    dropout: Optional[float]=None,
) -> nn.Sequential:
    layers = []
    current_dim = input_dim
    if initial_layer_norm:
        layers.append(nn.LayerNorm(current_dim))

    for n_features in features:
        layers.append(nn.Linear(current_dim, n_features))
        nn.init.zeros_(layers[-1].bias)
        layers.append(get_activation_fn(activation_fn))
        if activation_fn.lower().endswith("glu"):
            current_dim = n_features // 2
        else:
            current_dim = n_features
        if dropout is not None:
            layers.append(nn.Dropout(dropout))

    layers.append(nn.Linear(current_dim, output_dim))
    nn.init.zeros_(layers[-1].bias)
    if dropout is not None:
        layers.append(nn.Dropout(dropout))
    if final_activation_fn is not None:
        layers.append(get_activation_fn(final_activation_fn))

    if residual:
        return Residual(nn.Sequential(*layers))
    return nn.Sequential(*layers)


def masked_huber(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor], delta: float=1.0) -> torch.Tensor:
    diff = pred - target
    abs_diff = diff.abs()
    quad = torch.clamp(abs_diff, max=delta)
    lin = abs_diff - quad
    loss = 0.5 * quad * quad + delta * lin
    if mask is not None:
        if mask.dtype != torch.bool:
            mask = (mask != 0)
        mask = mask.unsqueeze(-1).to(loss.dtype)
        loss = loss * mask
        denom = mask.sum()
        if denom.item() == 0:
            return loss.sum() * 0.0
        return loss.sum() / denom
    return loss.mean()


@torch.no_grad()
def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, p=2, dim=-1)
    b = F.normalize(b, p=2, dim=-1)
    sim = torch.einsum('bsd,bkd->bsk', a, b)
    return sim


@torch.no_grad()
def greedy_match(sim: torch.Tensor, target_valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
    B, S, K = sim.shape
    if target_valid_mask is None:
        target_valid_mask = torch.ones(B, K, dtype=torch.bool, device=sim.device)
    else:
        target_valid_mask = target_valid_mask.to(sim.device)
        if target_valid_mask.dtype != torch.bool:
            target_valid_mask = (target_valid_mask != 0)
    
    matched_idx = torch.full((B, S), -1, dtype=torch.long, device=sim.device)
    matched_sim = torch.full((B, S), float("-inf"), dtype=sim.dtype, device=sim.device)
    available = target_valid_mask.clone()
    
    for s in range(S):
        cur = sim[:, s, :].clone()
        cur[~available] = float("-inf")
        best_k = cur.argmax(dim=-1)
        best_v = cur.gather(1, best_k.unsqueeze(1)).squeeze(1)
        can = torch.isfinite(best_v)
        matched_idx[can, s] = best_k[can]
        matched_sim[can, s] = best_v[can]
        if can.any():
            b_idx = torch.nonzero(can, as_tuple=False).squeeze(1)
            k_sel = best_k[can]
            available[b_idx, k_sel] = False
    return matched_idx, matched_sim


@torch.no_grad()
def mutual_nearest(sim: torch.Tensor, matched_idx: torch.Tensor, slot_valid: torch.Tensor, target_valid: torch.Tensor) -> torch.Tensor:
    B, S, K = sim.shape
    keep = torch.zeros(B, S, dtype=torch.bool, device=sim.device)
    sim_slots = sim.clone()
    sim_slots[~slot_valid.unsqueeze(-1).expand_as(sim_slots)] = float("-inf")
    best_slot_for_target = sim_slots.argmax(dim=1)
    for s in range(S):
        k = matched_idx[:, s]
        valid_row = slot_valid[:, s] & (k >= 0)
        if valid_row.any():
            b_idx = torch.nonzero(valid_row, as_tuple=False).squeeze(-1)
            k_sel = k[valid_row]
            target_ok = target_valid[b_idx, k_sel]
            slot_ok = (best_slot_for_target[b_idx, k_sel] == s)
            keep[b_idx] = keep[b_idx] | (target_ok & slot_ok)
    return keep


@torch.no_grad()
def align_targets(
    basis_slots: torch.Tensor,      # (B, S, D)
    target_slots: torch.Tensor,     # (B, K, D)
    target_exists: Optional[torch.Tensor],  # (B, K) in [0, 1] or bool.
    sim_threshold: float=0.35,
) -> Dict[str, torch.Tensor]:
    """
    Align basis_slots & target_slots by 1:1 greedy, generate masks.
    
    Returns:
        target_slots_aligned (B, S, D)
        target_mask: (B, S) in [0, 1] (float).
        matched_idx: (B, S) long
        matched_sim: (B, S) float
        sim_matrix: (B, S, K)
    """
    device = basis_slots.device
    B, S, D = basis_slots.shape
    _, K, _ = target_slots.shape
    
    sim = cosine_sim(basis_slots, target_slots)         # (B, S, K)
    m_idx, m_sim = greedy_match(sim, target_exists)     # (B, S), (B, S)
    
    keep = (m_idx >= 0) & (m_sim >= sim_threshold)      # (B, S)
    
    aligned = torch.zeros_like(basis_slots)
    gather_idx = m_idx.clamp_min(0).unsqueeze(-1).expand(B, S, D)   # (B, S, D)
    for b in range(B):
        aligned[b] = target_slots[b].gather(0, gather_idx[b])
    target_mask = keep.to(basis_slots.dtype)
    
    return {
        "target_slots_aligned": aligned,
        "target_mask": target_mask,
        "matched_idx": m_idx,
        "matched_sim": m_sim,
        "sim_matrix": sim,
    }


def psnr(x: torch.Tensor, y: torch.Tensor, eps: float=1e-8) -> torch.Tensor:
    if x.dim() == 3:
        x = x.unsqueeze(0)
    if y.dim() == 3:
        y = y.unsqueeze(0)
    
    x = x.float()
    y = y.float()
    
    mse = F.mse_loss(x, y, reduction='none')
    mse = mse.flatten(1).mean(dim=1)
    
    psnr = 10.0 * torch.log10(torch.tensor(1.0, device=x.device) / (mse+eps))
    return psnr
