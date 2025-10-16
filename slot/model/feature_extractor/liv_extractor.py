import torch
import torch.nn as nn
from liv import load_liv


class LIVImageFeatureExtractor(nn.Module):
    def __init__(
        self,
        model_type: str='resnet50',
        freeze: bool=True,
    ):
        super().__init__()
        
        model_type = model_type or 'resnet50'
        liv = load_liv(modelid=model_type)
        self.model = liv.module
        self.freeze = freeze
        
        if self.freeze:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)
    
    @property
    def out_dim(self) -> int:
        return self.model.output_dim
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if self.freeze:
            with torch.no_grad():
                return self.model(image)
        else:
            return self.model(image)