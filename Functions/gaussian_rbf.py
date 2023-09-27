import torch
import torch.nn as nn

#rbf kernel = exp[(-||x - x'||^2) / (2*(sigma)^2)]
class Gaussian_RBF(nn.Module):
    def __init__(self, num_funcs: int, cutoff: float, start: float = 0.0):
        super().__init__()
        self.rbf = num_funcs
        #offset = x'
        #create tensor of num_funcs equally spaced values from start to cutoff
        offset = torch.linspace(start, cutoff, num_funcs)
        #offset_width = sigma
        #create tensor the size of offset tensor with all values being the absolute distance between two values in offset
        offset_width = torch.FloatTensor(torch.abs(offset[0] - offset[1])*torch.ones_like(offset))
        self.register_buffer("offsets", offset)
        self.register_buffer("offset_width", offset_width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        #inputs[...,None] is done for broadcasting purposes to create extra dimension for compatibility iwth offsets
        numerator = torch.neg(torch.pow((inputs[...,None] - self.offsets), 2))
        denominator = 2 * torch.pow(self.offset_width, 2)
        return torch.exp(numerator / denominator)
