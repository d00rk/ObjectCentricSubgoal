from typing import Optional, List, Callable, Union
import torch
import torch.nn as nn
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

    layers.append(nn.Linear(current_dim, output_dim))
    nn.init.zeros_(layers[-1].bias)
    if final_activation_fn is not None:
        layers.append(get_activation_fn(final_activation_fn))

    if residual:
        return Residual(nn.Sequential(*layers))
    return nn.Sequential(*layers)