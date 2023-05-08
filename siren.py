import torch
from torch import nn
import numpy as np

class SineLayer(nn.Module):    
    def __init__(self, in_features, out_features, omega_0=20):
        super().__init__()
        self.omega_0 = omega_0
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, input):
        return torch.relu(self.omega_0 * self.linear(input))


class Siren(nn.Module):
    def __init__(self, device, in_features, hidden_features, hidden_layers, out_features, omega_0=1):
        super().__init__()
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, omega_0=omega_0))
        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, omega_0=omega_0))
#         self.net.append(SineLayer(hidden_features, out_features, 
#                                   omega_0=omega_0))
        final_linear = nn.Linear(hidden_features, out_features)
        self.net.append(final_linear)
        self.net = nn.Sequential(*self.net)
        self.device = device
        self.to(device)
    
    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output

def get_mgrid(sidelen, dim=2):
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid
