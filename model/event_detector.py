import torch
import torch.nn as nn

class EventDetector(nn.Module):
    """
    Temporal encoder over recent window of slots to output eventness p_t.
    Z_{t-k:t} -> sigmoid probability.
    """
    def __init__(self,
                 slot_dim: int=128,
                 num_slots: int=8,
                 window: int=6,
                 hidden_dim: int=256,
                 ):
        super().__init__()
        self.window = window
        in_dim = slot_dim * num_slots * window
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        # x: (B, window, S, D)
        B, W, S, D = x.shape
        x = x.reshape(B, -1)
        logits = self.net(x)
        return logits.squeeze(-1)   # (B, )
    