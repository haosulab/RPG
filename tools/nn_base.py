import torch
from torch import nn
import numpy as np

from tools.config import Configurable

def intprod(x):
    return int(np.prod(x))

def concat(args, axis=0):
    if isinstance(args[0], np.ndarray):
        return np.concatenate(args, axis=axis)
    else:
        return torch.concat(args, axis=axis)

def stack(args, axis=0):
    if isinstance(args[0], np.ndarray):
        return np.stack(args, axis=axis)
    else:
        return torch.stack(args, axis=axis)

class Network(Configurable, nn.Module):
    def __init__(self, cfg=None):
        Configurable.__init__(self, cfg)
        nn.Module.__init__(self)
        self._device = None

    def to(self, device):
        self._device = device
        return nn.Module.to(self, device)

    def cuda(self, device='cuda:0'):
        self._device = device
        return nn.Module.cuda(self, device)

    @property
    def device(self):
        #return next(self.parameters()).device
        if self._device is not None:
            return self._device
        else:
            self._device = next(self.parameters()).device
            return self._device

    def batch_input(self, x):
        from .utils import batch_input
        return batch_input(x, self.device)



class MySequential(nn.Sequential):
    def forward(self, *args, **kwargs):
        for idx, module in enumerate(self):
            if idx == 0:
                input = module(*args, **kwargs)
            else:
                input = module(input)
        return input