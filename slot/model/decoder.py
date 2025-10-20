from typing import Callable, Optional, Union, Tuple, Sequence
import random
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from slot.utils.neural_networks import get_activation_fn, build_grid_of_positions, build_transformer_encoder
from slot.utils.resizing import resize_patches_to_image


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


class PatchDecoder(nn.Module):
    """Decoder that takes object representations and reconstructs patches.

    Args:
        object_dim: Dimension of objects representations.
        output_dim: Dimension of each patch.
        num_patches: Number of patches P to reconstruct.
        decoder: Function that returns backbone to use for decoding. Function takes input and output
            dimensions and should return module that takes inputs of shape (B * K), P, N, and produce
            outputs of shape (B * K), P, M, where K is the number of objects, N is the number of
            input dimensions and M the number of output dimensions.
        decoder_input_dim: Input dimension to decoder backbone. If specified, a linear
            transformation from object to decoder dimension is added. If not specified, the object
            dimension is used and no linear transform is added.
        resize_mode: Type of upsampling for masks and targets.
        conditioned: Whether to condition the decoder on additional information (point or lang).
        top_k: Number of slots to decode per-position. Selects the top-k slots according to `mask`.
            Can also be a tuple that defines an interval [top_k_min, top_k_max] from which the top-k
            value is sampled.
        sampled_top_k: If true, the top-k slots are sampled from a categorical distribution without
            replacement, where the probabilities per-slot and position are specified by `mask`.
            If false, use hard top-k selection.
        training_top_k: Whether to apply top-k selection at training time.
        eval_top_k: Whether to apply top-k selection at evaluation time.
    """

    def __init__(
        self,
        object_dim: int,
        output_dim: int,
        num_patches: int,
        decoder: Callable[[int, int], nn.Module],
        decoder_input_dim: Optional[int] = None,
        upsample_target: Optional[float] = None,
        resize_mode: str = "bilinear",
        conditioned: bool = False,
        top_k: Optional[Union[int, Tuple[int]]] = None,
        sampled_top_k: bool = False,
        training_top_k: bool = False,
        eval_top_k: bool = False,
        pos_embed_scale: float = 0.02,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.num_patches = num_patches
        self.upsample_target = upsample_target
        self.resize_mode = resize_mode
        self.training_top_k = training_top_k
        self.eval_top_k = eval_top_k
        self.sampled_top_k = sampled_top_k
        if isinstance(top_k, Sequence):
            self.top_k_min, self.top_k_max = top_k
            assert self.top_k_min <= self.top_k_max
        else:
            self.top_k_min = self.top_k_max = top_k
        if (self.training_top_k or self.eval_top_k) and self.top_k_min is None:
            raise ValueError("Need to specify `top_k` if `training_top_k` or `eval_top_k` are set")
        if self.eval_top_k:
            self.top_k_eval = int((self.top_k_min + self.top_k_max) // 2)
        else:
            self.top_k_eval = None

        if decoder_input_dim is not None:
            self.inp_transform = nn.Linear(object_dim, decoder_input_dim, bias=True)
            nn.init.xavier_uniform_(self.inp_transform.weight)
            nn.init.zeros_(self.inp_transform.bias)
        else:
            self.inp_transform = None
            decoder_input_dim = object_dim
        if conditioned:
            self.merge_condition_info = build_transformer_encoder(
                decoder_input_dim, decoder_input_dim, 3, 4
            )
        self.conditioned = conditioned
        self.decoder = decoder(decoder_input_dim, output_dim + 1)
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches, decoder_input_dim) * pos_embed_scale
        )

    def get_top_k(self, num_objects: int) -> int:
        """Return the number of top-k slots to select."""
        if self.training:
            if self.top_k_min != self.top_k_max:
                top_k = random.randint(self.top_k_min, self.top_k_max)
            else:
                assert self.top_k_min is not None
                top_k = self.top_k_min
        else:
            top_k = self.top_k_eval
        return min(top_k, num_objects)

    def select_top_k(
        self, object_features: torch.Tensor, masks: torch.Tensor, k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select top-k objects per position according to their values in masks."""
        # object_features: [batch_dims] x n_objects x n_positions x dims
        # masks: [batch_dims] x n_objects x n_positions

        # Flatten batch dimensions
        batch_dims = object_features.shape[:-3]
        object_features = object_features.flatten(0, -4)
        batch_size, _, _, dims = object_features.shape

        with torch.no_grad():
            masks = masks.detach().flatten(0, -3)
            masks = einops.rearrange(masks, "b s p -> (b p) s")
            if self.training and self.sampled_top_k:
                idxs = torch.multinomial(masks, num_samples=k, replacement=False)
            else:
                idxs = torch.topk(masks, k=k, dim=1, sorted=False).indices
            idxs = einops.repeat(idxs, "(b p) k -> b k p d", b=batch_size, d=dims)

        object_features = torch.gather(object_features, dim=1, index=idxs)
        object_features = object_features.unflatten(0, batch_dims)
        idxs = idxs.unflatten(0, batch_dims)

        return object_features, idxs

    def restore_masks_after_top_k(
        self, masks: torch.Tensor, idxs: torch.Tensor, n_masks: int
    ) -> torch.Tensor:
        """Fill masks with zeros for all non-top-k objects."""
        # masks: [batch_dims] x top_k_objects x n_positions
        # idxs: [batch_dims] x top_k_objects x n_positions x dims
        batch_dims = masks.shape[:-2]
        masks_all = torch.zeros(*batch_dims, n_masks, masks.shape[-1], device=masks.device)
        masks_all.scatter_(dim=1, index=idxs[..., 0], src=masks)
        return masks_all

    def forward(
        self,
        object_features: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        patch_indices: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        condition_info: Optional[torch.Tensor] = None,
    ):
        assert object_features.dim() >= 3  # Image or video data.
        assert self.conditioned == (condition_info is not None)
        if self.upsample_target is not None and target is not None:
            target = (
                resize_patches_to_image(
                    target.detach().transpose(-2, -1),
                    scale_factor=self.upsample_target,
                    resize_mode=self.resize_mode,
                )
                .flatten(-2, -1)
                .transpose(-2, -1)
            )

        initial_shape = object_features.shape[:-1]
        num_objects = initial_shape[-1]
        object_features = object_features.flatten(0, -2)

        if self.inp_transform is not None:
            object_features = self.inp_transform(object_features)

        object_features = object_features.unsqueeze(1).expand(-1, self.num_patches, -1)

        # Simple learned additive embedding as in ViT
        object_features = object_features + self.pos_embed

        should_do_top_k = (self.training and self.training_top_k) or (
            not self.training and self.eval_top_k
        )
        if should_do_top_k:
            if masks is None:
                raise ValueError("Need to pass `masks` for top_k.")
            object_features, top_k_idxs = self.select_top_k(
                object_features.unflatten(0, initial_shape), masks, self.get_top_k(num_objects)
            )
            initial_shape = object_features.shape[:-1]
            object_features = object_features.flatten(0, -2)

        if patch_indices is not None:
            # Repeat indices for all slots
            indices = patch_indices.repeat_interleave(
                repeats=len(object_features) // len(patch_indices), dim=0
            )
            # Select only the positions specified by `indices`
            indices = indices.unsqueeze(-1).expand(-1, -1, object_features.shape[-1])
            object_features = object_features.gather(dim=1, index=indices)

        if condition_info is not None:
            condition_info = condition_info.reshape(object_features.shape[0], -1).unsqueeze(1)
            object_features = torch.cat((condition_info, object_features), dim=1)
            object_features = self.merge_condition_info(object_features)
            object_features = object_features[:, 1:]

        output = self.decoder(object_features)
        output = output.unflatten(0, initial_shape)

        # Split out alpha channel and normalize over slots.
        decoded_patches, alpha = output.split([self.output_dim, 1], dim=-1)

        masks = alpha.softmax(dim=-3)
        per_slot_patches = decoded_patches * masks
        reconstruction = torch.sum(decoded_patches * masks, dim=-3)
        masks = masks.squeeze(-1)

        if should_do_top_k:
            masks = self.restore_masks_after_top_k(masks, top_k_idxs, num_objects)

        if image is not None:
            if patch_indices is not None:
                # Create expanded mask with zeros where positions were not decoded.
                masks_to_resize = torch.zeros(*initial_shape, self.num_patches, device=masks.device)
                if len(initial_shape) == 3:  # Video case
                    indices = patch_indices.unsqueeze(1).unsqueeze(2)
                    indices = indices.expand(-1, masks.shape[1], masks.shape[2], -1)
                else:  # Image case
                    indices = patch_indices.unsqueeze(1).expand(-1, masks.shape[1], -1)
                masks_to_resize.scatter_(dim=-1, index=indices, src=masks)
            else:
                masks_to_resize = masks
            masks_as_image = resize_patches_to_image(
                masks_to_resize, size=image.shape[-1], resize_mode="bilinear"
            )
        else:
            masks_as_image = None

        output = {
            'reconstruction': reconstruction,
            'masks': masks,
            'masks_as_image': masks_as_image,
            'target': target if target is not None else None,
            'per_slot_patches': per_slot_patches
        }
        return output
    
 
class MLPDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int,
        activation_func: Union[str, Callable]='relu',
        final_activation_func: Optional[Union[str, Callable]]=None,
        initial_layer_norm: bool=False,
        residual: bool=False,
    ):
        super().__init__()
        
        layers = []
        current_dim = input_dim
        if initial_layer_norm:
            layers.append(nn.LayerNorm(current_dim))
        
        for _ in range(n_layers-1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(get_activation_fn(activation_func))
            if activation_func.lower().endswith('glu'):
                current_dim = hidden_dim // 2
            else:
                current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, output_dim))
        if final_activation_func is not None:
            layers.append(get_activation_fn(final_activation_func))
        
        self.net = nn.Sequential(*layers)
        self.residual = residual

        self._initial_weight()
        
    def _initial_weight(self):
        modules = list(self.net)
        L = len(modules)

        last_linear_idx = max(i for i, m in enumerate(modules) if isinstance(m, nn.Linear))

        def init_linear(m: nn.Linear, act: nn.Module | None, is_last: bool):
            if is_last and self.residual:
                nn.init.zeros_((m.weight))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                return

            if isinstance(act, (nn.ReLU, nn.GELU, nn.SiLU, nn.ELU, nn.SELU)):
                nn.init.kaiming_uniform_(m.weight, a=0.0, nonlinearity='relu')
            elif isinstance(act, nn.LeakyReLU):
                nn.init.kaiming_uniform_(m.weight, a=act.negative_slope, nonlinearity='leaky_relu')
            else:
                nn.init.xavier_uniform_(m.weight)

            if m.bias is not None:
                nn.init.zeros_(m.bias)

        for i, m in enumerate(modules):
            if not isinstance(m, nn.Linear):
                continue
            
            next_act = None
            for j in range(i + 1, L):
                if isinstance(modules[j], nn.Linear):
                    break
                if isinstance(modules[j],
                    (nn.ReLU, nn.LeakyReLU, nn.GELU, nn.SiLU, nn.ELU, nn.SELU, nn.GLU, nn.Tanh, nn.Sigmoid)):
                    next_act = modules[j]
                    break

            init_linear(m, next_act, i == last_linear_idx)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inputs = x
        x = self.net(x)
        if self.residual:
            x = inputs + x
        return x