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
    
    @property
    def out_dim(self) -> int:
        return self.model.outdim
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if self.freeze:
            with torch.no_grad():
                return self.model(image)
        else:
            return self.model(image)
        
        
        
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
#     image_features = ext(image)
#     # image_features /= image_features.norm(dim=-1, keepdim=True)
#     print(image_features.shape)