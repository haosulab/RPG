import torch
from torch import nn 
from .density_estimator import DensityEstimator
from tools.nn_base import Network
from tools.utils import logger


class RNDNet(Network):
    def __init__(self, inp_dim, cfg=None, n_layers=3, dim=512):
        super(RNDNet, self).__init__()
        self.inp_dim = inp_dim
        layers = []
        for i in range(n_layers):
            if i > 0:
                layers.append(nn.LeakyReLU())
            layers.append(nn.Linear(inp_dim, dim))
            inp_dim = dim
        self.main = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.main(x)

        
class RND(DensityEstimator):
    def __init__(self, space, cfg=None, normalizer='ema') -> None:
        super().__init__(space)

        self.target = RNDNet(self.inp_dim)
        for param in self.target.parameters():
            param.requires_grad = False

    def make_network(self, space):
        #return super().make_network(space)
        assert len(space.shape) == 1
        self.inp_dim = space.shape[0]
        network = RNDNet(self.inp_dim)
        return network

    def compute_loss(self, samples):
        from tools.utils import totensor
        inps = totensor(samples, device='cuda:0')
        predict = self.network(inps)
        with torch.no_grad():
            target = self.target(inps)
        loss = ((predict - target)**2).sum(axis=-1, keepdim=True)
        return loss

    def _update(self, samples):
        loss = self.compute_loss(samples)
        self.optimize(loss.mean())
        logger.logkv_mean(self.name + '_loss', loss.mean())

        return -loss.detach()

    def _log_prob(self, samples):
        return -self.compute_loss(samples) # negative as the log prob ..