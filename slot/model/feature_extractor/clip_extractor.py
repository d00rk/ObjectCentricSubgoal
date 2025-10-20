from typing import Union
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from slot.model.feature_extractor.base import BaseImageFeatureExtractor


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class CLIPImageFeatureExtractor(BaseImageFeatureExtractor):
    def __init__(
        self,
        model_type: str,        # 'RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'
        freeze: bool=True,
        device: Union[str, torch.device]=torch.device('cuda:0'),
    ):
        super().__init__()
        assert model_type in ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'], \
            f"Model type {model_type} is not supported for clip."
        
        self.freeze = freeze
        self.device = device
        if isinstance(device, str):
            self.device = torch.device(device)
        
        model, preprocess = clip.load(model_type, device=self.device)
        self.model = model.visual
        self.preprocess = preprocess
        self.resolution = getattr(self.model, 'input_resolution', 224)
        
        if self.freeze:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)
                
        mean = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(CLIP_STD).view(1, 3, 1, 1)
        self.register_buffer('_mean', mean, persistent=False)
        self.register_buffer('_std', std, persistent=False)
        
        self._spatial_fmap = None
        self._layer4_hook = None
        if hasattr(self.model, 'layer4'):
            def _save_fmap(module, inputs, output):
                self._spatial_fmap = output
            self._layer4_hook = self.model.layer4.register_forward_hook(_save_fmap)
                
    @property
    def dtype(self):
        return self.model.conv1.weight.dtype
        
    @property
    def out_dim(self) -> int:
        return self.model.output_dim
    
    def _preprocess_image(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(0)      # (1, C, H, W)
        assert x.ndim == 4 and x.shape[1] == 3, f"Expected BCHW with 3 channels, got {tuple(x.shape)}"
        
        if x.dtype == torch.uint8:
            x = x.float().div_(255.0)
        elif not torch.is_floating_point(x):
            x = x.float()
            
        if x.shape[-2] != self.resolution or x.shape[-1] != self.resolution:
            x = F.interpolate(x, size=(self.resolution, self.resolution), mode='bicubic', align_corners=False)
            
        mean = self._mean.to(device=x.device, dtype=x.dtype)
        std = self._std.to(device=x.device, dtype=x.dtype)
        x = (x - mean) / std
        
        x = x.to(dtype=self.dtype)
        return x
    
    def _forward_vit_tokens(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.conv1(x)                         # (B, width, grid, grid)
        x = x.reshape(x.shape[0], x.shape[1], -1)       # (B, width, grid*grid)
        x = x.permute(0, 2, 1)                          # (B, grid*grid, width)
        
        x = torch.cat([self.model.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)     # (B, grid*grid+1, width)
        x = x + self.model.positional_embedding.to(x.dtype)
        x = self.model.ln_pre(x)
        x = x.permute(1, 0, 2)              # (grid*grid+1, B, width)
        x = self.model.transformer(x)
        x = x.permute(1, 0, 2)              # (B, grid*grid+1, width)
        
        patch_tokens = x[:, 1:, :]
        cls_token = x[:, 0, :]
        if hasattr(self.model, 'ln_post'):
            cls_token = self.model.ln_post(cls_token)
        if self.model.proj is not None:
            global_feat = cls_token @ self.model.proj
        else:
            global_feat = cls_token
        
        return global_feat, patch_tokens

    def forward(self, image: torch.Tensor, return_spatial: bool=False, return_tokens: bool=False) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            image = self._preprocess_image(image)
        elif isinstance(image, Image.Image):
            image = self.preprocess(image)
            if image.ndim == 3:
                image = image.unsqueeze(0)
        else:
            raise ValueError(f"Input image must be torch.Tensor or PIL.Image, not {type(image)}")
        
        image = image.to(device=self.device, dtype=self.dtype)
        
        is_vit = hasattr(self.model, 'transformer') and hasattr(self.model, 'class_embedding')
        if self.freeze:
            with torch.no_grad():
                if is_vit:
                    global_feat, tokens = self._forward_vit_tokens(image)
                    if not (return_spatial or return_tokens):
                        return global_feat
                    if return_tokens and not return_spatial:
                        return global_feat, tokens
                    if return_spatial:
                        return (global_feat, None, tokens) if return_tokens else (global_feat, None)
                else:
                    global_feat = self.model(image)
                    if not (return_spatial or return_tokens):
                        return global_feat
                    fmap = self._spatial_fmap
                    assert fmap is not None
                    if return_tokens:
                        B, C, H, W = fmap.shape
                        tokens = fmap.flatten(2).transpose(1, 2).contiguous()
                        return (global_feat, fmap, tokens) if return_spatial else (global_feat, tokens)
                    else:
                        return global_feat, fmap
        else:
            if is_vit:
                global_feat, tokens = self._forward_vit_tokens(image)
                if not (return_spatial or return_tokens):
                    return global_feat
                if return_tokens and not return_spatial:
                    return global_feat, tokens
                if return_spatial:
                    return (global_feat, None, tokens) if return_tokens else (global_feat, None)
            else:
                global_feat = self.model(image)
                if not (return_spatial or return_tokens):
                    return global_feat
                fmap = self._spatial_fmap
                assert fmap is not None
                if return_tokens:
                    B, C, H, W = fmap.shape
                    tokens = fmap.flatten(2).transpose(1, 2).contiguous()
                    return (global_feat, fmap, tokens) if return_spatial else (global_feat, tokens)
                else:
                    return global_feat, fmap


class CLIPTextModel(nn.Module):
    def __init__(
        self,
        model_type: str,
        freeze: bool=True,
        device: Union[torch.device, str]=torch.device('cuda:0'),
    ):
        super().__init__()
        
        assert model_type in ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'], \
            f"Model type {model_type} is not supported for clip."
        
        self.device = device
        if isinstance(device, str):
            self.device = torch.device(device)
        self.freeze = freeze
        
        model, preprocess = clip.load(model_type, device=self.device)
        self._dtype = model.dtype
        self.model = model
        
        if self.freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)
    
    @property
    def dtype(self):
        return self._dtype
    
    @property
    def out_dim(self) -> int:
        return 512
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.freeze:
            with torch.no_grad():
                output = self.model.encode_text(tokens)
            return output
        else:
            output = self.model.encode_text(tokens)
            return output
            
    
    
    
if __name__=='__main__':
    from PIL import Image
    import numpy as np
    device = torch.device('cuda:0')
    extractor = CLIPImageFeatureExtractor(model_type='ViT-B/32')     # 대략 1380MiB 정도
    extractor = extractor.to(device)
    dtype = extractor.dtype
    print(extractor.out_dim)
    image = 'libero_test.png'
    image = Image.open(image).convert('RGB')
    image_features, fmap, tokens = extractor(image, return_tokens=True, return_spatial=True)
    print('feature:', image_features.shape)
    print('tokens: ', tokens.shape)
    # print('fmap: ', fmap.shape)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    
    img = 'libero_test_1.png'
    img = Image.open(img).convert('RGB')
    img = extractor.preprocess(img).unsqueeze(0).to(device=device, dtype=dtype)
    img_features = extractor(img)
    img_features /= img_features.norm(dim=-1, keepdim=True)
    
    text = 'a photo of cat.'
    text = clip.tokenize(text)
    text = text.to(device)
    text_encoder = CLIPTextModel(model_type='ViT-B/32')
    text_encoder = text_encoder.to(device)
    text_feature = text_encoder(text)
    print(text_feature.shape)
    text_feature /= text_feature.norm(dim=-1, keepdim=True)
    
    similarity0 = (100.0 * image_features @ text_feature.T).item()
    print(similarity0)
    
    similarity1 = (100.0 * img_features @ text_feature.T).item()
    print(similarity1)