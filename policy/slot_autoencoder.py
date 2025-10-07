from typing import List, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.slot_modules import (SoftPositionEmbed, SlotAttention, Conv2d_NHWC,
                                spatial_broadcast, spatial_flatten, unstack_and_split)


class SlotAttentionAutoEncoder(nn.Module):
    def __init__(self,
                 resolution,
                 num_slots,
                 num_iterations):
        super().__init__()
        self.resolution = resolution
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        
        self.encoder_cnn = nn.Sequential(
            Conv2d_NHWC(3, 64, kernel_size=5, padding='same', activation=F.relu),
            Conv2d_NHWC(64, 64, kernel_size=5, padding='same', activation=F.relu),
            Conv2d_NHWC(64, 64, kernel_size=5, padding='same', activation=F.relu),
            Conv2d_NHWC(64, 64, kernel_size=5, padding='same', activation=F.relu)
        )
        
        self.decoder_initial_size = (8, 8)
        self.decoder_cnn = nn.Sequential(
            Conv2d_NHWC(64, 64, kernel_size=5, stride=2, padding='same', activation=F.relu, transpose=True),
            Conv2d_NHWC(64, 64, kernel_size=5, stride=2, padding='same', activation=F.relu, transpose=True),
            Conv2d_NHWC(64, 64, kernel_size=5, stride=2, padding='same', activation=F.relu, transpose=True),
            Conv2d_NHWC(64, 64, kernel_size=5, stride=2, padding='same', activation=F.relu, transpose=True),
            Conv2d_NHWC(64, 64, kernel_size=5, stride=1, padding='same', activation=F.relu, transpose=True),
            Conv2d_NHWC(64, 4, kernel_size=3, stride=1, padding='same', activation=None, transpose=True)
        )
        
        self.encoder_pos = SoftPositionEmbed(64, self.resolution)
        self.decoder_pos = SoftPositionEmbed(64, self.decoder_initial_size)
        
        self.layer_norm = nn.LayerNorm(64)
        self.mlp = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        self.slot_attention = SlotAttention(
            num_iterations=self.num_iterations,
            num_slots=self.num_slots,
            slot_size=64,
            mlp_hidden_size=128
        )
        
    def forward(self, image):
        """
        image: (B, H, W, C)
        """
        x = self.encoder_cnn(image)
        x = self.encoder_pos(x)
        x = spatial_flatten(x)
        x = self.mlp(self.layer_norm(x))        # (B, W*H, D)
        
        slots = self.slot_attention(x)
        
        x = spatial_broadcast(slots, self.decoder_initial_size)
        
        x = self.decoder_pos(x)
        x = self.decoder_cnn(x)                 # (B*S, W, H, C+1)
        
        recons, masks = unstack_and_split(x, batch_size=image.shape[0])
        masks = F.softmax(masks, dim=1)
        recon_combined = torch.sum(recons * masks, dim=1)
        
        return recon_combined, recons, masks, slots
    
    def encode_slots(self, image):
        # image: (B, H, W, C)
        x = self.encoder_cnn(image)
        x = self.encoder_pos(x)
        x = spatial_flatten(x)      
        x = self.mlp(self.layer_norm(x))    # (B, H*W, D)
        slots = self.slot_attention(x)      # (B, S, D)
        return slots
        
        
class SlotAttentionClassifier(nn.Module):
    def __init__(self,
                 resolution,
                 num_slots,
                 num_iterations):
        super().__init__()
        self.resolution = resolution
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        
        self.encoder_cnn = nn.Seqeuntial(
            Conv2d_NHWC(3, 64, kernel_size=5, padding='same', activation='relu'),
            Conv2d_NHWC(64, 64, kernel_size=5, stride=2, padding='same', activation=F.relu),
            Conv2d_NHWC(64, 64, kernel_size=5, stride=2, padding='same', activation=F.relu),
            Conv2d_NHWC(64, 64, kernel_size=5, padding='same', activation=F.relu)
        )
        self.encoder_pos = SoftPositionEmbed(64, (32, 32))
        
        self.layer_norm = nn.LayerNorm(64)
        self.mlp = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        self.slot_attention = SlotAttention(
            num_iterations=self.num_iterations,
            num_slots=self.num_slots,
            slot_size=64,
            mlp_hidden_size=128
        )
        
        self.mlp_classifier = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 19),
            nn.Sigmoid()
        )
        
    def forward(self, image):
        image = image.permute(0, 2, 3, 1).contiguous()      # B C H W -> B H W C
        x = self.encoder_cnn(image)
        x = self.encoder_pos(x)
        x = spatial_flatten(x)
        x = self.mlp(self.layer_norm(x))
        
        slots = self.slot_attention(x)
        
        predictions = self.mlp_classifier(slots)
        
        return predictions