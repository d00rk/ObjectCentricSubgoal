from typing import Tuple, List
import torch
import torch.nn as nn

from policy.slot_autoencoder import SlotAttentionAutoEncoder

class SlotEncoder(nn.Module):
    def __init__(self,
                 in_channels: int=3,
                 hidden_dim: int=128,
                 num_slots: int=8,
                 slot_dim: int=64,
                 num_iterations: int=3,
                 mlp_hidden_dim: int=128,
                 resolution: List=[128,128],
                 ):
        super().__init__()
        self.slot_dim = slot_dim
        self.num_slots = num_slots
        self.autoencoder = SlotAttentionAutoEncoder(resolution=resolution, num_slots=num_slots, num_iterations=num_iterations)
        
    def forward(self, x):
        # x: (B, T, C, H, W) -> NHWC per frame batch
        B, T, C, H, W = x.shape
        x = x.permute(0, 1, 3, 4, 2).contiguous().view(B*T, H, W, C)
        slots = self.autoencoder.encode_slots(x)        # (B*T, S, D)
        S, D = slots.shape[1], slots.shape[2]
        slots = slots.view(B, T, S, D)
        return slots
    

class PositionalEncoding(nn.Module):
    def __init__(self,
                 d_model: int,
                 max_len: int=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)            # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x, t_offset: int=0):
        # x: (B, T, S, D)
        B, T, S, D = x.shape
        return x + self.pe[:, t_offset:t_offset+T, :D].unsqueeze(2)
    
class SlotFormer(nn.Module):
    """
    Transformer over (time, slot) tokens.
    Predicts Z_{t+1} given Z_{1:t}
    Teacher-forced during training.
    """
    def __init__(self,
                 slot_dim: int=128,
                 num_layers: int=4,
                 num_heads: int=4,
                 ff_dim: int=512,
                 dropout: float=0.1,
                 ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=slot_dim,
                                                   nhead=num_heads,
                                                   dim_feedforward=ff_dim,
                                                   dropout=dropout,
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.posenc = PositionalEncoding(slot_dim)
        self.pred_head = nn.Linear(slot_dim, slot_dim)
    
    def forward(self, x):
        # x: (B, T, S, D)
        B, T, S, D = x.shape
        x = self.posenc(x)
        tokens = x.reshape(B, -1, D)
        h = self.transformer(tokens)
        h = h.reshape(B, T, S, D)
        h_last = h[:, -1]               # one-step prediction using last time step features
        pred = self.pred_head(h_last)   # (B, S, D)
        return pred