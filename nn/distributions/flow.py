import torch
from torch.distributions import Categorical
from tools.utils import batched_index_select

from .dist_head import ActionDistr
from .dist_head import DistHead
from torch import nn

# import pyro.distributions as dist
# import pyro.distributions.transforms as T


class FlowAction(ActionDistr):
    def __init__(self, flow_dist, batch_size=1):
        # NOTE: flow is conditioned ..
        self.flow_dist = flow_dist
        self.batch_size = batch_size

    def rsample(self, sample_shape=None, *args, **kwargs):
        if sample_shape is None:
            sample_shape = (self.batch_size,)
        samples = self.flow_dist.rsample(sample_shape)
        log_prob = self.flow_dist.log_prob(samples)
        return samples, log_prob

    def sample(self):
        raise NotImplementedError
        samples = self.flow_dist.sample((self.batch_size,))
        log_prob = self.flow_dist.log_prob(samples.detach())
        #print(samples.shape, log_prob.shape)
        return samples, log_prob

    def log_prob(self, action, sum=True):
        raise NotImplementedError
        assert action.shape[0] == self.batch_size
        assert sum
        return self.flow_dist.log_prob(action)

    def get_parameters(self):
        return next(self.flow_dist.parameters())

    def entropy(self, n=1):
        return None
        samples = self.flow_dist.sample((n,))
        return self.flow_dist.log_prob(samples).mean()

    def probs(self, action, sum=True):
        raise NotImplementedError

        
class Flow(DistHead):
    def __init__(self, action_space, cfg = None, layers=4, hidden_dim=128, tanh=True, mode='affine_autoregressive', linear=None, std_mode=None, squash=None, std_scale=None):
        super().__init__(action_space)

        self.base_mean = nn.Parameter(torch.zeros(self.action_dim), requires_grad=False)
        self.base_std = nn.Parameter(torch.ones(self.action_dim), requires_grad=False)
        self.base_dist = dist.Normal(self.base_mean, self.base_std)
        flows = []
        for i in range(layers):
            if mode == 'affine_autoregressive':
                flows.append(T.conditional_affine_autoregressive(self.action_dim, hidden_dim, hidden_dims=[128, 128]))
            else:
                flows.append(T.conditional_spline_autoregressive(self.action_dim, hidden_dim))
        self.flows = torch.nn.ModuleList(flows)
        if tanh:
            flows = flows + [T.TanhTransform()] #, T.AffineTransform(self.action_bias, self.action_scale)]
        self.conditional_dist = dist.ConditionalTransformedDistribution(self.base_dist, flows)

    def get_input_dim(self):
        return self._cfg.hidden_dim

    def forward(self, cond):
        dist = FlowAction(self.conditional_dist.condition(cond), batch_size=len(cond))
        dist.device = self.device
        return dist

        
if __name__ == '__main__':
    from solver.envs import TrajBandit
    from rl.agent import Actor

    env = TrajBandit()

    actor = Actor(env.observation_space, env.action_space, head=dict(TYPE='Flow')).cuda()
    obs = env.reset()

    action = actor(obs)
    print(action.rsample()[0].device, obs.device)