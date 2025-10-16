import torch

class BaseImageFeatureExtractor(torch.nn.Module):
    @property
    def out_dim(self):
        pass