import torch
import torch.nn as nn
import numpy as np

class shifted_softplus(nn.Module):
    #ssp(x) = ln(0.5*e^x + 0.5) = ln(1 + e^x) - ln(2)
    def __init__(self):
        super().__init__()
        self.shift = np.log(2.0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return nn.functional.softplus(input) - self.shift