import torch
import torch.nn as nn
from .builder import build_from_cfg

ACTIVATION_LAYERS = {}

for module in [nn.LeakyReLU, nn.ReLU, nn.Sigmoid, nn.Softplus, nn.Softsign, nn.Tanh, nn.Threshold,
               nn.Softmin, nn.Softmax, nn.Softmax2d, nn.LogSoftmax, nn.AdaptiveLogSoftmaxWithLoss]:
    ACTIVATION_LAYERS[module.__name__] = module


class Clamp(nn.Module):
    def __init__(self, min=-1., max=1.):
        super(Clamp, self).__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        return torch.clamp(x, min=self.min, max=self.max)


def build_activation_layer(cfg):
    """Build activation layer.
    Args:
        cfg (dict): The activation layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate an activation layer.
    Returns:
        nn.Module: Created activation layer.
    """
    return build_from_cfg(cfg, ACTIVATION_LAYERS)