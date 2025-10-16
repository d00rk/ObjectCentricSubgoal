from typing import Optional
import torch
import torch.nn as nn

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


def build_grid_of_positions(resolution, min=0.0):
    ranges = [torch.linspace(min, 1.0, steps=res) for res in resolution]
    grid = torch.meshgrid(*ranges, indexing='ij')
    grid = torch.stack(grid, dim=-1)
    grid = torch.reshape(grid, [resolution[0], resolution[1], -1])
    return grid