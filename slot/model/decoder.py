from typing import Callable, Optional, Union
import torch
import torch.nn as nn
from slot.utils.neural_networks import get_activation_fn, build_grid_of_positions


def get_slotattention_decoder_backbone(object_dim: int, output_dim: int = 4):
    """Get CNN decoder backbone form the original slot attention paper."""
    return nn.Sequential(
        nn.ConvTranspose2d(object_dim, 64, 5, stride=2, padding=2, output_padding=1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(64, 64, 5, stride=2, padding=2, output_padding=1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(64, 64, 5, stride=2, padding=2, output_padding=1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(64, 64, 5, stride=2, padding=2, output_padding=1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(64, 64, 5, stride=1, padding=2, output_padding=0),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(64, output_dim, 3, stride=1, padding=1, output_padding=0),
    )


class SlotAttentionDecoder(nn.Module):
    def __init__(
        self,
        decoder: nn.Module,
        final_activation: Union[str, Callable] = 'identity',
        positional_embedding: Optional[nn.Module]=None,
    ):
        super().__init__()
        
        self.initial_conv_size = (8, 8)
        self.decoder = decoder
        self.final_activation = get_activation_fn(final_activation)
        self.positional_embedding = positional_embedding
        if positional_embedding:
            self.register_buffer('grid', build_grid_of_positions(self.initial_conv_size))
        
    def forward(
        self,
        object_features: torch.Tensor
    ) -> torch.Tensor:
        assert object_features.dim() >= 3       # image or video
        initial_shape = object_features.shape[:-1]
        object_features = object_features.flatten(0, -2)
        
        object_features = (
            object_features.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, *self.initial_conv_size)
        )
        if self.positional_embedding:
            object_features = self.positional_embedding(object_features, self.grid.unsqueeze(0))
        
        output = self.decoder(object_features)
        output = output.unflatten(0, initial_shape)
        
        rgb, alpha = output.split([3, 1], dim=-3)
        rgb = self.final_activation(rgb)
        alpha = alpha.softmax(dim=-4)
        
        return {
            'reconstruction': (rgb*alpha).sum(-4),
            'object_reconstructions': rgb,
            'masks': alpha.squeeze(-3),
        }