import torch
import torch.nn as nn
from vip import load_vip
from slot.model.feature_extractor.base import BaseImageFeatureExtractor


class VIPImageFeatureExtractor(BaseImageFeatureExtractor):
    def __init__(
        self,
        model_type: str='resnet50',
        freeze: bool=True,
    ):
        super().__init__()
        
        assert 'resnet' in model_type, f"{model_type} is not supported for VIP. Available model type: ['resnet18', 'resnet34', 'resnet50']"
        
        vip = load_vip(model_type)
        self.model = vip.module
        self.freeze = freeze
        
        if self.freeze:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)
    
    @property
    def out_dim(self) -> int:
        if self.model.hidden_dim > 0:
            return self.model.hidden_dim
        return self.model.outdim
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if self.freeze:
            with torch.no_grad():
                return self.model(image)
        else:
            return self.model(image)