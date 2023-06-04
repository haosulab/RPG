import numpy as np
import torch
import torch.nn as nn
import torch.distributions as thdist
import torch.nn.functional as thfunc

from normal_std import Std


class TanhNormalHead(nn.Module):

    def __init__(self, out_dim, std_cfg=dict()):
        super().__init__()
        self.std = Std(out_dim, **std_cfg)
        self.out_dim = out_dim
        self.in_dim = self.out_dim + self.std.in_dim
    
    def forward(self, x):
        assert x.shape[-1] == self.in_dim
        mu = x[..., :self.out_dim]
        std = self.std(x[..., self.out_dim:])
        # sample and log_prob
        dist = thdist.Normal(loc=mu, scale=std)
        u = dist.rsample()
        a = torch.tanh(u)
        log_prob_a = dist.log_prob(u) - (2 * (np.log(2) - u - thfunc.softplus(-2 * u)))
        return dict(dist=dist, a=a, log_prob_a=log_prob_a)
