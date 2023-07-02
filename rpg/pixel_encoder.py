import torch
import numpy as np
from torch import nn
from tools.utils import Seq

class NormalizeImg(nn.Module):
    """Normalizes pixel observations to [0,1) range."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.float().div(255.)




class Flatten(nn.Module):
    """Flattens its input to a (batched) vector."""
    def __init__(self):
        super().__init__()
	
    def forward(self, x):
        return x.view(x.size(0), -1)


def _get_out_shape(in_shape, layers):
    """Utility function. Returns the output shape of a network for a given input shape."""
    x = torch.randn(*in_shape).unsqueeze(0)
    return (nn.Sequential(*layers) if isinstance(layers, list) else layers)(x).squeeze(0).shape


def make_cnn(inp_shape, latent_dim, num_channels):
    #C = int(3*cfg.frame_stack)
    C = inp_shape[0]
    layers = [NormalizeImg(),
                nn.Conv2d(C, num_channels, 7, stride=2), nn.ReLU(),
                nn.Conv2d(num_channels, num_channels, 5, stride=2), nn.ReLU(),
                nn.Conv2d(num_channels, num_channels, 3, stride=2), nn.ReLU(),
                nn.Conv2d(num_channels, num_channels, 3, stride=2), nn.ReLU()]

    out_shape = _get_out_shape(inp_shape, layers)
    from tools.utils import mlp
    layers.extend([Flatten(), mlp(np.prod(out_shape), 256, latent_dim)]) #nn.Linear(np.prod(out_shape), latent_dim)])
    return Seq(*layers)