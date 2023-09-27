import torch
import torch.nn as nn
import numpy as np

#Creates strength reduction as distance increases between nodes
class Cosine_Cutoff(nn.Module):
    def __init__(self, cutoff: float = 5.0 ):
        super().__init__()
        self.register_buffer("cutoff", torch.FloatTensor([cutoff]))

    def forward(self, input: torch.Tensor, cutoff: torch.Tensor) -> torch.Tensor:
        #create cosine cutoff barrier
        new_input = 0.5 * (torch.cos(input * np.pi / cutoff) + 1.0)
        #remove elements beyond barrier: if less than cutoff, multiply by True -> float -> 1, else multiply by 0 (erase)
        new_input *= (input < cutoff).float()
        return new_input