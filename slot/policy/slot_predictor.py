from typing import Optional, Tuple, Dict, List, Union
import os
import math
import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip


class SlotSubgoalPredictor(nn.Module):
    """
    Slot Subgoal Predictor.
    
    - Visual: R3M/CLIP encoder (frozen) -> Mapping g (freeze or training) -> query image space
    - Text: CLIP text encoder (frozen)
    - Slot Attention: instruction-conditioned object slots (freeze or training)
    - Predictor: subgoal slot predictor (training)
    - Decoder: query feature reconstruction from slots (freeze or training)
    - RGB Decoder (Optional): reconstructs rgb image from slots (training)
    """
    def __init__(
        self,
        predictor: nn.Module,
        pretrained_ckpt_path: str,
        rgb_decoder: Optional[nn.Module]=None,
        freeze_mapping: bool=True,
        freeze_slot_attention: bool=True,
        freeze_decoder: bool=True,
    ):
        super().__init__()

        self.predictor = predictor
        self.rgb_decoder = rgb_decoder
        self._initialize_modules(pretrained_ckpt_path)
        self._load_ckpt(pretrained_ckpt_path)
        
        self.freeze_mapping = freeze_mapping
        self.freeze_slot_attention = freeze_slot_attention
        self.freeze_decoder = freeze_decoder
        
        if freeze_mapping:
            self.mapping.eval()
            for p in self.mapping.parameters():
                p.requires_grad_(False)
        if freeze_slot_attention:
            self.slot_attention.eval()
            for p in self.slot_attention.parameters():
                p.requires_grad_(False)
        if freeze_decoder:
            self.decoder.eval()
            for p in self.decoder.parameters():
                p.requires_grad_(False)
    
    def _initialize_modules(self, ckpt_path):
        if ckpt_path is None or ckpt_path == "" or not isinstance(ckpt_path, str):
            raise ValueError("Provide correct checkpoint path.")
        
        if not os.path.isfile(ckpt_path):
            raise ValueError(f"Checkpoint path not found: {ckpt_path}")
        
        raw = torch.load(ckpt_path, map_location='cpu')
        cfg = raw['cfg']
        del raw
        
        visual_encoder_cfg = cfg['model'].get("visual_encoder", None)
        print(visual_encoder_cfg)
        text_encoder_cfg = cfg['model'].get("text_encoder", None)
        mapping_cfg = cfg['model'].get("mapping", None)
        slot_attention_cfg = cfg['model'].get("slot_attention", None)
        decoder_cfg = cfg['model'].get("decoder", None)
        
        self.visual_encoder = hydra.utils.instantiate(visual_encoder_cfg)
        self.text_encoder = hydra.utils.instantiate(text_encoder_cfg)
        self.mapping = hydra.utils.instantiate(mapping_cfg)
        self.slot_attention = hydra.utils.instantiate(slot_attention_cfg)
        self.decoder = hydra.utils.instantiate(decoder_cfg)
        
        
    def _load_ckpt(self, ckpt_path: str):
        if ckpt_path is None or ckpt_path == "" or not isinstance(ckpt_path, str):
            raise ValueError("Provide correct checkpoint path.")
        
        if not os.path.isfile(ckpt_path):
            raise ValueError(f"Checkpoint path not found: {ckpt_path}")
        
        raw = torch.load(ckpt_path, map_location='cpu')
        if isinstance(raw, dict):
            if isinstance(raw.get("state_dict", None), dict):
                sd = raw['state_dict']
            elif isinstance(raw.get("model", None), dict):
                sd = raw["model"]
            else:
                sd = raw
        else:
            raise ValueError("Invalid checkpoint format.")
        
        roots = ["", "module.", "model.", "net.", "encoder.", "slot_encoder."]
        target_names = ["visual_encoder", "text_encoder", "mapping", "slot_attention", "decoder"]
        
        def _select_subdict(sd, candidates):
            for pref in candidates:
                picked = {k[len(pref):]: v for k, v in sd.items() if k.startswith(pref)}
                if len(picked) > 0:
                    return picked, pref
            return {}, ""
        
        for name in target_names:
            submod = getattr(self, name, None)
            if submod is None:
                print(f"Predictor has no submodule '{name}'.")
                continue
            
            candidates = [f'{r}{name}.' for r in roots]
            picked, used_prefix = _select_subdict(sd, candidates)
            
            if len(picked) == 0:
                extra_candidates = []
                for k in sd.keys():
                    pos = k.find(f"{name}.")
                    if pos > 0:
                        extra_candidates.append(k[:pos+len(f"{name}.")])
                extra_candidates = sorted(set(extra_candidates), key=len)
                if extra_candidates:
                    picked, used_prefix = _select_subdict(sd, extra_candidates)
            
            if len(picked) == 0:
                print(f"No weights for submodule '{name}' found in ckpt")
                continue
            
            sub_sd = submod.state_dict()
            filtered = {k: v for k, v in picked.items() if k in sub_sd}
            
            if len(filtered) == 0:
                print(f"'{name}' has no matching keys after filtering.")
                continue
            
            missing, unexpected = submod.load_state_dict(filtered, strict=True)
            print(f"Loaded weights for {name}")

    def encode(self, images: torch.Tensor, insturction: str) -> Dict[str, torch.Tensor]:
        B, C, H, W = images.shape
        device = images.device
        dtype = images.dtype
        if dtype == torch.uint8:
            images = images.float().div_(255.0)
            dtype = images.dtype
        if images.max() >= 5.0:
            images = images.div_(255.0)
        
        visual_features, visual_tokens = self.visual_encoder(images, return_tokens=True)
        visual_tokens = visual_tokens.to(device=device, dtype=dtype)
        _, P, _ = visual_tokens.shape
        patch = int(math.sqrt(P))
        
        text_tokens = clip.tokenize(insturction, truncate=True).to(device, non_blocking=True)
        text_features = self.text_encoder(text_tokens)
        text_features = text_features.to(dtype=dtype)
        text_features = F.normalize(text_features, dim=-1)
        
        mapped_visual_tokens = self.mapping(visual_tokens)
        mapped_visual_tokens_normalized = F.normalize(mapped_visual_tokens, dim=-1)
        
        slot, attn, attn_list = self.slot_attention(
            inputs=mapped_visual_tokens_normalized,
            text_embeddings=text_features.unsqueeze(1)
        )
        
        return {
            'slots': slot,
            'attn': attn,
            'visual_tokens': visual_tokens.view(B, patch, patch, -1),
            'mapped_visual_tokens': mapped_visual_tokens,
            'text_features': text_features
        }
    
    def decode(self, slots: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = slots.shape[0]
        
        recon = self.decoder(slots)
        pred = recon['reconstruction']
        num_patches = pred.shape[1]
        patch = int(math.sqrt(num_patches))
        pred = pred.view(B, patch, patch, -1)
        
        return {
            'reconstruction': pred,
            'masks': recon['masks'],
            'masks_as_image': recon['masks_as_image'],
            'per_slot_patches': recon['per_slot_patches']
        }
    
    def decode_rgb(self, slots: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.rgb_decoder is None:
            return None
        return self.rgb_decoder(slots)
    
    def forward(self, images: torch.Tensor, instructions: str) -> Dict[str, torch.Tensor]:
        encode_out = self.encode(images, instructions)
        slots = encode_out['slots']
        slot_attn = encode_out['attn']
        img_tokens = encode_out['visual_tokens']
        mapped_tokens = encode_out['mapped_visual_tokens']
        text_features = encode_out['text_features']
        
        pred_out = self.predictor(slots=slots, lang_embeddings=text_features)
        pred_slots = pred_out['slots']
        
        decode_out = self.decode(pred_slots)
        pred_tokens = decode_out['reconstruction']
        pred_masks = decode_out['masks']
        pred_per_slot_patches = decode_out['per_slot_patches']
        
        return {
            'current_slots': slots,
            'current_slots_attn': slot_attn,
            'current_image_tokens': img_tokens,
            'current_mapped_image_tokens': mapped_tokens,
            'pred_slots': pred_slots,
            'pred_image_tokens': pred_tokens,
            'pred_token_masks': pred_masks,
            'pred_per_slott_patches': pred_per_slot_patches,
            'text_features': text_features
        }