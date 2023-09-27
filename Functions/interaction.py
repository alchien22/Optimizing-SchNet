import torch.nn as nn
from torch.nn import Linear, Dense
from Functions import shifted_softplus

class Interaction(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = Linear()
        self.ssp = shifted_softplus()

        filters, filters, none
        rbf, filters, ssp
        filters, features, ssp
        features, features, none
        features, filters, none
    def forward():
        return