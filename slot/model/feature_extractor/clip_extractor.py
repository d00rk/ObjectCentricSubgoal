import torch
import torch.nn as nn
import clip
from slot.model.feature_extractor.base import BaseImageFeatureExtractor


class CLIPImageFeatureExtractor(BaseImageFeatureExtractor):
    def __init__(
        self,
        model_type: str,        # 'RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'
        freeze: bool=True,
        device: torch.device=torch.device('cuda:0'),
    ):
        super().__init__()
        assert model_type in ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'], \
            f"Model type {model_type} is not supported for clip."
        
        self.freeze = freeze
        self.device = device
        
        model, preprocess = clip.load(model_type, device=self.device)
        self.model = model.visual
        self.preprocess = preprocess
        
        if self.freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)
                
    @property
    def dtype(self):
        return self.model.conv1.weight.dtype
        
    @property
    def out_dim(self) -> int:
        return self.model.output_dim
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if self.freeze:
            with torch.no_grad():
                return self.model(image)
        else:
            return self.model(image)


class CLIPTextModel(nn.Module):
    def __init__(
        self,
        model_type: str,
        freeze: bool=True,
        device: torch.device=torch.device('cuda:0'),
    ):
        super().__init__()
        
        assert model_type in ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'], \
            f"Model type {model_type} is not supported for clip."
            
        self.device = device
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
            
    
    
    
# if __name__=='__main__':
#     from PIL import Image
#     import numpy as np
#     device = torch.device('cuda:0')
    # extractor = CLIPImageFeatureExtractor(model_type='ViT-B/32')     # 대략 1380MiB 정도
    # extractor = extractor.to(device)
    # dtype = extractor.dtype

    # image = 'libero_test.png'
    # image = Image.open(image).convert('RGB')
    # image = extractor.preprocess(image).unsqueeze(0).to(device=device, dtype=dtype)
    # image_features = extractor(image)
    # image_features /= image_features.norm(dim=-1, keepdim=True)
    
    # img = 'libero_test_1.png'
    # img = Image.open(img).convert('RGB')
    # img = extractor.preprocess(img).unsqueeze(0).to(device=device, dtype=dtype)
    # img_features = extractor(img)
    # img_features /= img_features.norm(dim=-1, keepdim=True)
    
    # text = 'a photo of cat.'
    # text = clip.tokenize(text)
    # text = text.to(device)
    # text_encoder = CLIPTextModel(model_type='ViT-B/32')
    # text_encoder = text_encoder.to(device)
    # text_feature = text_encoder(text)
    # print(text_feature.shape)
    # text_feature /= text_feature.norm(dim=-1, keepdim=True)
    
    # similarity0 = (100.0 * image_features @ text_feature.T).item()
    # print(similarity0)
    
    # similarity1 = (100.0 * img_features @ text_feature.T).item()
    # print(similarity1)