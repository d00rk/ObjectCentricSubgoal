from typing import List, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SlotAttention(nn.Module):
    def __init__(self,
                 num_iterations,
                 num_slots: int,
                 slot_size: int,
                 mlp_hidden_size: int,
                 epsilon: float=1e-8,
                 ):
        super().__init__()
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.epsilon = epsilon
        
        self.norm_inputs = nn.LayerNorm(self.slot_size)
        self.norm_slots = nn.LayerNorm(self.slot_size)
        self.norm_mlp = nn.LayerNorm(self.slot_size)
        
        self.slots_mu = nn.Parameter(torch.zeros(1, 1, self.slot_size))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, 1, self.slot_size))
        
        self.project_q = nn.Linear(self.slot_size, self.slot_size, bias=False)
        self.project_k = nn.Linear(self.slot_size, self.slot_size, bias=False)
        self.project_v = nn.Linear(self.slot_size, self.slot_size, bias=False)
        
        self.gru = nn.GRUCell(self.slot_size, self.slot_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.slot_size, self.mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.mlp_hidden_size, self.slot_size)
        )
        
        self._init_parameters()
    
    def _init_parameters(self):
        for m in [self.project_q, self.project_k, self.project_v]:
            nn.init.xavier_uniform_(m.weight)
        nn.init.xavier_uniform_(self.slots_mu)
        nn.init.xavier_uniform_(self.slots_log_sigma)
        
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, inputs):
        """
        inputs: (B, N, D) tokens
        return: (B, S, D) slots
        """
        B, N, D = inputs.shape
        inputs = self.norm_inputs(inputs)
        k = self.project_k(inputs)          # B, N, D
        v = self.project_v(inputs)          # B, N, D
        
        mu = self.slots_mu.expand(B, self.num_slots, -1)
        sigma = torch.exp(self.slots_log_sigma).expand(B, self.num_slots, -1)
        slots = mu + sigma * torch.randn_like(mu)
        
        for _ in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)
            
            q = self.project_q(slots)
            q = q * (self.slot_size ** -0.5)
            attn_logits = torch.bmm(q, k.transpose(1, 2))
            attn = F.softmax(attn_logits, dim=-1)
            
            attn = attn + self.epsilon
            attn = attn / attn.sum(dim=-1, keepdim=True)
            updates = torch.bmm(attn, v)
            
            slots = self._gru_update(updates, slots_prev)
            slots += self.mlp(self.norm_mlp(slots))
        return slots
    
    def _gru_update(self, updates, slots_prev):
        B, S, D = updates.shape
        updates = updates.reshape(B*S, D)
        slots_prev = slots_prev.reshape(B*S, D)
        new = self.gru(updates, slots_prev)
        return new.reshape(B, S, D)

def spatial_broadcast(slots, resolution):
    """
    Broadcast slot features to a 2D grid and collapse slot dimension.
    
    Args:
        - slots: (B, S, D)
        - resolution: (W, H)
    Returns:
        - grid: (B*S, W, H, D)
    """
    B, S, D = slots.shape
    W, H = resolution
    x = slots.reshape(B*S, 1, 1, D)
    x = x.expand(B*S, W, H, D)
    return x

def spatial_flatten(x):
    """
    Flatten spatial dimensions.
    
    Args:
        - x: (B, W, H, C)
    Returns:
        - x: (B, W*H, C)
    """
    B, W, H, C = x.shape
    return x.reshape(B, W*H, C)

def unstack_and_split(x, batch_size, num_channels: int=3):
    """
    Unstack batch dimension and split into channels and alpha mask.
    
    Args:
        - x: (B*S, W, H, num_channels+1)
        - batch_size: B
    Returns:
        - channels: (B, S, W, H, num_channels)
        - masks: (B, S, W, H, 1)
    """
    BS, W, H, C_all = x.shape
    S = BS // batch_size
    unstacked = x.reshape(batch_size, S, W, H, C_all)
    channels, masks = torch.split(unstacked, [num_channels, 1], dim=-1)
    return channels, masks

class Conv2d_NHWC(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding='same',
                 activation=None,
                 transpose=False):
        super().__init__()
        self.transpose = transpose
        self.activation = activation
        if padding == 'same':
            pad = kernel_size // 2 if isinstance(kernel_size, int) else kernel_size[0] // 2
        else:
            pad = 0
            
        if not transpose:
            self.op = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad)
        else:
            outpad = 1 if (stride == 2) else 0
            self.op = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, output_padding=outpad)
            
    def forward(self, x):
        # x: (B, H, W, C)
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        x = self.op(x)
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        if self.activation is not  None:
            x = self.activation(x)
        return x
        
def build_gird(resolution):
    W, H = resolution
    ranges = [np.linspace(0.0, 1.0, num=res, dtype=np.float32) for res in (W, H)]
    grid = np.meshgrid(*ranges, sparse=False, indexing='ij')
    grid = np.stack(grid, axis=-1)      # (W, H, 2)
    grid = np.reshape(grid, [1, W, H, 2]) # (1, W, H, 2)
    grid_comp = np.concatenate([grid, 1.0-grid], axis=-1)
    return torch.from_numpy(grid_comp.astype(np.float32))


class SoftPositionEmbed(nn.Module):
    def __init__(self,
                 hidden_size,
                 resolution):
        super().__init__()
        self.dense = nn.Linear(4, hidden_size, bias=True)
        grid = build_gird(resolution)
        self.register_buffer("grid", grid, persistent=False)
        
    def forward(self, inputs):
        # inputs: (B, W, H, C)
        B, W, H, C = inputs.shape
        grid = self.grid.to(device=inputs.device, dtype=inputs.dtype)   # (1, W, H, 4)
        pos = self.dense(grid)          # (1, W, H, C)
        pos = pos.expand(B, -1, -1, -1)
        return inputs + pos
