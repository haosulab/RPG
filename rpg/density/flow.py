from torch import nn
import torch
# import pyro.distributions as dist
# import pyro.distributions.transforms as T
from .density_estimator import DensityEstimator


class Flow(DensityEstimator):
    def __init__(self, action_space, cfg = None, layers=4, hidden_dim=128, tanh=False, mode='spline_autoregressive'):
        super().__init__(action_space)

    def make_network(self, space):
        inp_dim = space.shape[0]
        #hidden_dim = self._cfg.hidden_dim

        base_mean = nn.Parameter(torch.zeros(inp_dim), requires_grad=False).cuda()
        base_std = nn.Parameter(torch.ones(inp_dim), requires_grad=False).cuda()
        base_dist = dist.Normal(base_mean, base_std)

        flows = []
        for i in range(self._cfg.layers):
            if self._cfg.mode == 'affine_autoregressive':
                flows.append(T.affine_autoregressive(inp_dim, hidden_dims=[128, 128]))
            else:
                flows.append(T.spline_autoregressive(inp_dim, hidden_dims=[128, 128]))
        if self._cfg.tanh:
            flows = flows + [T.TanhTransform()] #, T.AffineTransform(self.action_bias, self.action_scale)]
        self.dist = dist.TransformedDistribution(base_dist, flows)
        return nn.ModuleList(flows)

    #def get_input_dim(self):
    #    return self._cfg.hidden_dim

    #def forward(self, cond):
    #    dist = FlowAction(self.conditional_dist.condition(cond), batch_size=len(cond))
    #    dist.device = self.device
    #    return dist

    def _log_prob(self, samples, log=False):
        return self.dist.log_prob(samples)[:, None]

    def _update(self, samples):
        logprob = self._log_prob(samples, log=True)
        self.optimize(-logprob.mean())
        return logprob.detach()