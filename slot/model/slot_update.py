import torch
import torch.nn as nn


class RecurrentGatedCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        input_gating: bool=True,
        hidden_gating: bool=True,
        update_transform: bool=True,
        convex_update: bool=False,
    ):
        super().__init__()
        
        self.input_gating = input_gating
        self.hidden_gating = hidden_gating
        self.update_transform = update_transform
        self.convex_update = convex_update
        
        if not input_gating and not hidden_gating:
            raise ValueError(f"At least one of 'input_gating' or 'hidden_gating' needs to be active.")
        
        if update_transform:
            if input_gating:
                self.to_update = nn.Linear(input_size, 2*hidden_size)
            else:
                self.to_update = nn.Linear(input_size, hidden_size)
        else:
            if input_size != hidden_size:
                raise ValueError(f"If 'update_transform==False', input_size must be equal to hidden_size.")
            if input_gating:
                self.to_update = nn.Linear(input_size, hidden_size)
            else:
                self.to_update = nn.Identity()
        
        if hidden_gating:
            self.gating_hidden = nn.Linear(hidden_size, hidden_size)
        else:
            self.gating_hidden = None
    
    def forward(self, input: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        if self.update_transform:
            if self.input_gating:
                update, gate = self.to_update(input).chunk(2, dim=-1)
            else:
                update = self.to_update(input)
                gate = torch.zeros_like(update)
        else:
            update = input
            if self.input_gating:
                gate = self.to_update(input)
            else:
                gate = torch.zeros_like(update)
        
        if self.hidden_gating:
            gate = gate + self.gating_hidden(hidden)
        
        gating = nn.functional.sigmoid(gate)
        if self.convex_update:
            return (1 - gating) * hidden + gating * update
        else:
            return hidden + gating * update