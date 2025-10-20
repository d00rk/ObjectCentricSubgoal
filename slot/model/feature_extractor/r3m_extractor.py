import torch
import torch.nn as nn
from r3m import load_r3m
from slot.model.feature_extractor.base import BaseImageFeatureExtractor


class R3MImageFeatureExtractor(BaseImageFeatureExtractor):
    def __init__(
        self,
        model_type: str='resnet50',
        freeze: bool=True,
    ):
        super().__init__()
        
        assert 'resnet' in model_type, f"Model name {model_type} is not supported for r3m. Valid choices: ['resnet18', 'resnet34', 'resnet50']"
            
        r3m = load_r3m(model_type)
        self.model = r3m.module
        self.freeze = freeze
        
        if self.freeze:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)
        
        self._spatial_fmap = None
        self._layer4_hook_handle = self.model.convnet.layer4.register_forward_hook(
            self._save_spatial_hook
        )
    
    @property
    def out_dim(self) -> int:
        return self.model.outdim
    
    def _save_spatial_hook(self, module, inputs, output):
        self._spatial_fmap = output
    
    def remove_hooks(self):
        if self._layer4_hook_handle is not None:
            self._layer4_hook_handle.remove()
            self._layer4_hook_handle = None
    
    def forward(self, image: torch.Tensor, return_spatial: bool=False, return_tokens: bool=False,) -> torch.Tensor:
        if self.freeze:
            with torch.no_grad():
                global_feat = self.model(image)
        else:
            global_feat = self.model(image)
        
        if not (return_spatial or return_tokens):
            return global_feat
        
        fmap = self._spatial_fmap
        assert fmap is not None
        
        if return_tokens:
            B, C, H, W = fmap.shape
            tokens = fmap.flatten(2).transpose(1, 2).contiguous()
            if return_spatial:
                return global_feat, fmap, tokens
            else:
                return global_feat, tokens
        else:
            return global_feat, fmap
        
# if __name__ == '__main__':
#     ext = R3MImageFeatureExtractor()
#     print(ext.out_dim)
#     # print(ext)
    
    
#     from PIL import Image
#     import numpy as np
#     device = torch.device('cuda:0')
#     image = 'libero_test.png'
#     image = Image.open(image).convert('RGB')
#     image = np.array(image)
#     image = torch.from_numpy(image)
#     image = image.permute(2, 0, 1).contiguous()
#     image = image.unsqueeze(0).to(device)
#     ext = ext.to(device)
#     # image = extractor.preprocess(image).unsqueeze(0).to(device=device, dtype=dtype)
#     global_feature, fmp, tokens = ext(image, return_spatial=True, return_tokens=True)
#     # image_features /= image_features.norm(dim=-1, keepdim=True)
#     print(global_feature.shape)
#     print(fmp.shape)
#     print(tokens.shape)