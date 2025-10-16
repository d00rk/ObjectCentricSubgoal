from typing import Dict, Optional, Union, Callable
import torch
import torch.nn as nn
from slot.utils.normalization import pixel_norm


class SlotAttention(nn.Module):
    def __init__(
        self,
        dim: int,                                                                   # dimension of each slot
        feature_dim: int,                                                           # dimension of features which is the output of feature extractor
        kvq_dim: Optional[int]=None,                                                # dimension of query, key, value after projection
        n_heads: int=1,
        n_iters: int=3,
        eps: float=1e-8,
        ff_mlp: Optional[Union[nn.Module, Callable[[], nn.Module]]]=None,
        slot_update: Optional[Union[nn.Module, Callable[[], nn.Module]]]=None,
        use_gru: bool=True,
        use_projection_bias: bool=False,
        use_implicit_differentiation: bool=False,
        use_cosine_attention: bool=False,
        use_input_norm: bool=True,
        dual_conditioning: bool=False,
        point_conditioning: bool=False,
        lang_conditioning: bool=False,
        use_query_init: bool=True,
        num_slots: Optional[int]=None,                                              # number of slots
        num_lang_slots: int=0,                                                      # number of slots to initialized with language embedding. must be equal or smaller than number of slots.
        text_dim: int=512,                                                          # dimension of text embedding.
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_iters = n_iters
        self.eps = eps
        
        self.use_implicit_differentiation = use_implicit_differentiation
        self.use_cosine_attention = use_cosine_attention
        
        # conditioning
        self.dual_conditioning = dual_conditioning
        self.point_conditioning = point_conditioning
        self.lang_conditioning = lang_conditioning
        
        # initialization of query
        self.use_query_init = use_query_init
        self.num_slots = num_slots
        self.num_lang_slots = num_lang_slots
        self.text_dim = text_dim
        if self.use_query_init:
            assert self.num_slots is not None and self.num_lang_slots > 0 and self.num_lang_slots <= self.num_slots, \
                f"Provide correct number of slots and number of language slots to use query initialization."
        
        if kvq_dim is None:
            self.kvq_dim = dim
        else:
            self.kvq_dim = kvq_dim
            
        assert self.kvq_dim % self.n_heads == 0, f"Key, value, query dimensions must be divisible by number of heads."
        self.dims_per_head = self.kvq_dim // self.n_heads
        self.scale = self.dims_per_head ** -0.5
        
        self.norm_slots = nn.LayerNorm(dim)
        
        qdim = dim
        if self.dual_conditioning:
            self.add_dim = 2 + 128
            qdim += self.add_dim
        elif self.point_conditioning:
            self.add_dim = 2
            qdim += self.add_dim
        elif self.lang_conditioning:
            self.add_dim = 256
            qdim += self.add_dim
            
        self.qdim = qdim
        
        self.to_q = nn.Linear(qdim, self.kvq_dim, bias=use_projection_bias)
        self.to_k = nn.Linear(feature_dim, self.kvq_dim, bias=use_projection_bias)
        self.to_v = nn.Linear(feature_dim, self.kvq_dim, bias=use_projection_bias)
        
        if use_gru:
            assert slot_update is None, f"'use_gru' and 'slot_update' are mutually exclusive."
            self.gru = nn.GRUCell(self.kvq_dim, dim)
        else:
            self.gru = None
            if isinstance(slot_update, nn.Module):
                self.to_update = slot_update
            elif callable(slot_update):
                self.to_update = slot_update()
            elif self.n_heads > 1 or self.kvq_dim != dim:
                self.to_update = nn.Linear(self.kvq_dim, dim, bias=use_projection_bias)
            else:
                self.to_update = nn.Identity()
        
        if use_input_norm:
            self.norm_input = nn.LayerNorm(feature_dim)
        else:
            self.norm_input = nn.Identity()
        
        if isinstance(ff_mlp, nn.Module):
            self.ff_mlp = ff_mlp
        elif callable(ff_mlp):
            self.ff_mlp = ff_mlp()
        elif ff_mlp is None:
            self.ff_mlp = nn.Identity()
        else:
            raise ValueError("ff_mlp needs to be nn.Module or callable returning nn.Module.")    
        
        if self.use_query_init:
            self.text_to_slot = nn.Linear(self.text_dim, dim)
            rem = self.num_slots - self.num_lang_slots
            self.learnable_slots = nn.Parameter(torch.randn(rem, dim))
    
    def step(
        self,
        slots,
        k,
        v,
        masks=None,
        point_conditioning=None,
        dual_conditioning=None,
        lang_conditioning=None,
    ):
        B, S, _ = slots.shape
        slots_prev = slots
        
        slots = self.norm_slots(slots)
        if self.dual_conditioning:
            if dual_conditioning is None:
                slots = torch.cat(
                    [slots, torch.zeros(B, S, self.add_dim).to(slots.device)], dim=-1
                )
            else:
                slots = torch.cat([slots, dual_conditioning], dim=-1)
        elif self.point_conditioning:
            if point_conditioning is None:
                slots = torch.cat(
                    [slots, torch.zeros(B, S, self.add_dim).to(slots.device)], dim=-1
                )
            else:
                slots = torch.cat([slots, point_conditioning], dim=-1)
        elif self.lang_conditioning:
            if lang_conditioning is None:
                slots = torch.cat(
                    [slots, torch.zeros(B, S, self.add_dim).to(slots.device)], dim=-1
                )
            else:
                slots = torch.cat([slots, lang_conditioning], dim=-1)
        
        q = self.to_q(slots).view(B, S, self.n_heads, self.dims_per_head)
        
        if self.use_cosine_attention:
            q = pixel_norm(q)
            k = pixel_norm(k)
        q = q * self.scale
        
        dots = torch.einsum("bihd,bjhd->bihj", q, k)
        if masks is not None:
            dots.masked_fill_(masks.to(torch.bool).view(B, S, 1, 1), float("-inf"))
        
        attn = dots.flatten(1, 2).softmax(dim=1)
        attn = attn.view(B, S, self.n_heads, -1)
        attn_before_reweighting = attn
        attn = attn / (attn.sum(dim=-1, keepdim=True) + self.eps)
        
        updates = torch.einsum("bjhd,bihj->bihd", v, attn).flatten(-2, -1)
        
        if self.gru is not None:
            slots = self.gru(updates.reshape(-1, self.kvq_dim), slots_prev.reshape(-1, self.dim))
            slots = slots.reshape(updates.shape[0], -1, self.dim)
        elif isinstance(self.to_update, (nn.Identity, nn.Linear)):
            slots = slots_prev + self.to_update(updates)
        else:
            slots = self.to_update(updates, slots_prev)
        
        if self.ff_mlp:
            slots = self.ff_mlp(slots)
        
        return slots, attn_before_reweighting.mean(dim=2)
    
    def iterate(
        self,
        n_iters,
        slots,
        k,
        v,
        masks=None,
        point_conditioning=None,
        dual_conditioning=None,
        lang_conditioning=None,
    ):
        attn = None
        attn_list = []
        for _ in range(n_iters):
            slots, attn = self.step(
                slots, k, v, masks, point_conditioning, dual_conditioning, lang_conditioning
            )
            attn_list.append(attn)
        return slots, attn_list
    
    def forward(
        self,
        inputs: torch.Tensor,
        conditioning: Optional[torch.Tensor],
        masks: Optional[torch.Tensor]=None,
        point_conditioning=None,
        dual_conditioning=None,
        lang_conditioning=None,
        text_embeddings: Optional[torch.Tensor]=None,
    ):
        B, N, D = inputs.shape
        
        if self.use_query_init and conditioning is None:
            assert text_embeddings is not None, f"'use_query_init' requires text_embeddings."
            if text_embeddings.dim() ==  2:
                text_embeddings = text_embeddings.unsqueeze(1)
            
            if text_embeddings.size(1) < self.num_lang_slots:
                pad = self.num_lang_slots - text_embeddings.size(1)
                text_embeddings = torch.cat(
                    [text_embeddings, text_embeddings[:, -1:].repeat(1, pad, 1)], dim=1
                )
            
            # M slots initialized by language embeddings.
            lang_slots = self.text_to_slot(text_embeddings[:, :self.num_lang_slots])        # (B, M, dim)
            
            # (K-M) slots
            learnable = self.learnable_slots.unsqueeze(0).repeat(B, 1, 1)                   # (B, K-M, dim)
            conditioning = torch.cat([lang_slots, learnable], dim=1)                        # (B, K, dim)
        
        slots = conditioning
        inputs = self.norm_input(inputs)
        k = self.to_k(inputs).view(B, N, self.n_heads, self.dims_per_head)
        v = self.to_v(inputs).view(B, N, self.n_heads, self.dims_per_head)
        
        if self.use_implicit_differentiation:
            slots, _, attn_list = self.iterate(
                self.n_iters - 1, slots, k, v, masks, point_conditioning, dual_conditioning, lang_conditioning
            )
            slots, attn = self.step(
                slots.detach(), k, v, masks, point_conditioning, dual_conditioning, lang_conditioning
            )
            attn_list.append(attn)
        else:
            slots, attn, attn_list = self.iterate(
                self.n_iters, slots, k, v, masks, point_conditioning, dual_conditioning, lang_conditioning
            )
        
        return slots, attn, attn_list
    
    
class SlotAttentionGrouping(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        object_dim: int,
        kvq_dim: Optional[int]=None,
        n_heads: int=1,
        n_blocks: int=1,
        n_iters: int=3,
        eps: float=1e-8,
        ff_mlp: Optional[Union[nn.Module, Callable[[], nn.Module]]]=None,
        slot_update: Optional[Union[nn.Module, Callable[[], nn.Module]]]=None,
        positional_embedding: Optional[nn.Module]=None,
        use_gru: bool=True,
        use_projection_bias: bool=False,
        use_implicit_differentiation: bool=False,
        use_cosine_attention: bool=False,
        use_empty_slot_for_masked_slots: bool=False,
    ):
        super().__init__()
        
        self._object_dim = object_dim
        
        def make_slot_attention(use_input_norm=True):
            return SlotAttention(
                dim=object,
                feature_dim=feature_dim,
                kvq_dim=kvq_dim,
                n_heads=n_heads,
                n_iters=n_iters,
                eps=eps,
                ff_mlp=ff_mlp,
                slot_update=slot_update,
                use_gru=use_gru,
                use_projection_bias=use_projection_bias,
                use_cosine_attention=use_cosine_attention,
                use_implicit_differentiation=use_implicit_differentiation,
                use_input_norm=use_input_norm,
            )
        
        if n_blocks == 1:
            self.slot_attention = make_slot_attention()
            self.norm_input = nn.Identity()
        else:
            self.slot_attention = nn.ModuleList(
                [make_slot_attention(use_input_norm=False) for _ in range(n_blocks)]
            )
            self.norm_input = nn.LayerNorm(feature_dim)
        
        self.positional_embedding = positional_embedding
        
        if use_empty_slot_for_masked_slots:
            self.empty_slot = nn.Parameter(torch.randn(object_dim) * object_dim**-0.5)
        else:
            self.empty_slot = None
            
    @property
    def object_dim(self):
        return self._object_dim
    
    def forward(
        self,
        feature: Dict[str, torch.Tensor],
        conditioning: Optional[torch.Tensor],       # (B, K, D_slot)
        slot_mask: Optional[torch.Tensor]=None,
        text_embeds: Optional[torch.Tensor]=None,   # (B, K, D_text)
    ) -> Dict[str, torch.Tensor]:
        if self.positional_embedding:
            features = self.positional_embedding(feature['features'], feature.get('positions', None))
        else:
            features = feature['features']
        
        features = self.norm_input(features)
        if isinstance(self.slot_attention, SlotAttention):
            slots, attn, _ = self.slot_attention(features, conditioning, slot_mask, text_embeddings=text_embeds)
        else:
            slots = conditioning
            for slot_att in self.slot_attention:
                slots, attn, _ = slot_att(features, slots, slot_mask, text_embeddings=text_embeds)
        
        if slot_mask is not None and self.empty_slot is not None:
            slots[slot_mask] = self.empty_slot.to(dtype=slots.dtype)
        
        return {
            "slots": slots,
            "feature_attributions": attn,
            "is_empty": slot_mask,
        }


class ControllableSlotAttentionGrouping(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        object_dim: int,
        kvq_dim: Optional[int]=None,
        n_heads: int=1,
        n_blocks: int=1,
        n_iters: int=3,
        eps: float=1e-8,
        ff_mlp: Optional[Union[nn.Module, Callable[[], nn.Module]]]=None,
        slot_update: Optional[Union[nn.Module, Callable[[], nn.Module]]]=None,
        positionial_embedding: Optional[nn.Module]=None,
        use_gru: bool=True,
        use_projection_bias: bool=False,
        use_implicit_differentation: bool=False,
        use_cosine_attention: bool=False,
        use_empty_slot_for_masked_slots: bool=False,
        point_conditioning: bool=False,
        dual_conditioning: bool=False,
        lang_conditioning: bool=False,
        use_query_init: bool=True,
        num_slots: Optional[int]=None,
        num_lang_slots: int=0,
        text_dim: int=512,
    ):
        super().__init__()
        self._object_dim = object_dim
        self.dual_conditioning = dual_conditioning
        self.point_conditioning = point_conditioning
        self.lang_conditioning = lang_conditioning
        
        def make_slot_attention(use_input_norm=True):
            return SlotAttention(
                dim=object_dim,
                feature_dim=feature_dim,
                kvq_dim=kvq_dim,
                n_head=n_heads,
                n_iters=n_iters,
                eps=eps,
                ff_mlp=ff_mlp,
                slot_update=slot_update,
                use_gru=use_gru,
                use_projection_bias=use_projection_bias,
                use_implicit_differentiation=use_implicit_differentation,
                use_cosine_attention=use_cosine_attention,
                use_input_norm=use_input_norm,
                dual_conditioning=dual_conditioning,
                point_conditioning=point_conditioning,
                lang_conditioning=lang_conditioning,
                use_query_init=use_query_init,
                num_slots=num_slots,
                num_lang_slots=num_lang_slots,
                text_dim=text_dim
            )
        
        if n_blocks == 1:
            self.slot_attention = make_slot_attention()
            self.norm_input = nn.Identity
        else:
            self.slot_attention = nn.ModuleList(
                [make_slot_attention(use_input_norm=False) for _ in range(n_blocks)]
            )
            self.norm_input = nn.LayerNorm(feature_dim)
        
        self.positional_embedding = positionial_embedding
        self.n_iters = n_iters
        if use_empty_slot_for_masked_slots:
            self.empty_slot = nn.Parameter(torch.randn(object_dim) * object_dim**-0.5)
        else:
            self.empty_slot = None
    
    @property
    def object_dim(self):
        return self._object_dim
    
    def forward(
        self,
        feature: Dict[str, torch.Tensor],
        conditioning: Optional[Dict[str, torch.Tensor]]=None,
        dual_conditioning: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]]=None,
        point_conditioning: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]]=None,
        lang_conditioning: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]]=None,
        slot_mask: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]]=None,
        text_embeddings: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]]=None,
    ) -> Dict[str, torch.Tensor]:
        
        if self.positional_embedding:
            features = self.positional_embedding(feature['features'], feature.get('positions', None))
        else:
            features = feature['features']
        
        features = self.norm_input(features)
        
        if isinstance(self.slot_attention, SlotAttention):
            if conditioning is not None or self.slot_attention.use_query_init:
                slots, attn, _ = self.slot_attention(features, conditioning, slot_mask, text_embeddings=text_embeddings)
            else:
                slots, attn = None, None
            
            if dual_conditioning is not None:
                if self.dual_conditioning:
                    dual_slots, dual_attn, dual_attn_list = self.slot_attention(
                        features, conditioning, slot_mask, dual_conditioning=dual_conditioning
                    )
                else:
                    dual_slots, dual_attn, dual_attn_list = self.slot_attention(
                        features, dual_conditioning, slot_mask
                    )
            else:
                dual_slots = None
                dual_attn = None
                dual_attn_list = None
            
            if point_conditioning is not None:
                if self.point_conditioning:
                    point_slots, point_attn = self.slot_attention(
                        features, conditioning, slot_mask, point_conditioning=point_conditioning
                    )
                else:
                    point_slots, point_attn = self.slot_attention(
                        features, point_conditioning, slot_mask
                    )
            else:
                point_slots = None
                point_attn = None
            
            if lang_conditioning is not None:
                if self.lang_conditioning:
                    lang_slots, lang_attn = self.slot_attention(
                        features, conditioning, slot_mask, lang_conditioning=lang_conditioning
                    )
                else:
                    lang_slots, lang_attn = self.slot_attention(
                        features, lang_conditioning, slot_mask
                    )
            else:
                lang_slots = None
                lang_attn = None
        else:
            slots = conditioning
            for slot_attn in self.slot_attention:
                slots, attn = slot_attn(features, slots, slot_mask, text_embeddings=text_embeddings)
            
            dual_slots = dual_conditioning
            for slot_attn in self.slot_attention:
                dual_slots, dual_attn_attn = slot_attn(features, dual_slots, slot_mask, text_embedding=text_embeddings)
            
            point_slots = point_conditioning
            for slot_attn in self.slot_attention:
                point_slots, point_attn = slot_attn(features, point_slots, slot_mask, text_embeddings=text_embeddings)
        
        if slot_mask is not None and self.empty_slot is not None:
            slots[slot_mask] = self.empty_slot.to(dtype=slots.dtype)
        
        out: Dict[str, torch.Tensor] = {
            'slots': slots,
            'dual_slots': dual_slots,
            'point_slots': point_slots,
            'lang_slots': lang_slots,
            'feature_attributions': attn,
            'dual_feature_attributions': dual_attn,
            'point_feature_attributions': point_attn,
            'lang_feature_attributions': lang_attn,
            'is_empty': slot_mask,
        }
        
        if dual_attn_list is None:
            return out
        else:
            out['dual_attn_1'] = dual_attn_list[0]
            out['dual_attn_2'] = dual_attn_list[1]
            out['dual_attn_3'] = dual_attn_list[2]
            return out


# class StickBreakingGrouping(nn.Module):
#     def __init__(
#         self,
#         object_dim: int,
#         feature_dim: int,
#         n_slots: int,
#         kernel_var: float=1.0,
#         learn_kernel_var: bool=False,
#         max_unexplained: float=0.0,
#         min_slot_mask: float=0.0,
#         min_max_mask_value: float=0.0,
#         early_termination: bool=False,
#         add_unexplained: bool=False,
#         eps: float=1e-8,
#         detach_features: bool=False,
#         use_input_layernorm: bool=False,
#     ):
#         super().__init__()
        
#         self.n_slots = n_slots
#         self.object_dim = object
        
#         assert kernel_var > 0.0
#         if learn_kernel_var:
#             self.kernel_logvar = nn.Parameter(torch.tensor(math.log(kernel_var)))
#         else:
#             self.register_buffer("kernel_logvar", torch.tensor(math.log(kernel_var)))
        
#         assert 0.0 <= max_unexplained < 1.0
#         self.max_unexplained = max_unexplained
        
#         assert 0.0 <= min_slot_mask < 1.0
#         self.min_slot_mask = min_slot_mask
        
#         assert 0.0 <= min_max_mask_value < 1.0
#         self.min_max_mask_value = min_max_mask_value
        
#         self.early_termination = early_termination
#         self.add_unexplained = add_unexplained
        
#         if add_unexplained and not early_termination:
#             raise ValueError("'add_unexplained=True' only works with 'early_termination=True'")