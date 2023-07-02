import torch
from torch import nn
from .dist_head import DistHead, ActionDistr
import numpy as np
import torch.nn.functional as F
from torch.distributions import Normal as Gaussian
# from .gmm import GMMAction


class NormalAction(ActionDistr):
    def __init__(self, loc, scale, tanh=False, act_scale=1.):
        self.dist = Gaussian(loc, scale)
        self.tanh = tanh
        self.kwargs = {'tanh': tanh, 'scale': act_scale}
        self.act_scale = act_scale

    def rsample(self, detach=False):
        action = self.dist.rsample()
        if detach:
            action = action.detach()

        logp = self.dist.log_prob(action)
        if self.tanh:
            # https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/core.py
            logp -= (2*(np.log(2) - action - F.softplus(-2*action)))
            action = torch.tanh(action)

        logp = logp.sum(axis=-1)
        return action * self.act_scale, logp

    def entropy(self):
        #TODO: sac entropy bugs here ..
        assert not self.tanh
        return self.dist.entropy().sum(axis=-1)

    def sample(self):
        return self.rsample(detach=True)

    def get_parameters(self):
        return self.dist.loc, self.dist.scale

    def log_prob(self, action):
        if self.tanh:
            raise NotImplementedError
        action = action / self.act_scale
        return self.dist.log_prob(action).sum(axis=-1)

    def render(self, print_fn):
        print_fn('loc:', self.dist.loc.detach().cpu().numpy(), 'std:', self.dist.scale.detach().cpu().numpy(), 'entropy:', self.dist.entropy().detach().cpu().numpy())


class Normal(DistHead):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -20

    STD_MODES = ['statewise', 'fix_learnable', 'fix_no_grad']

    def __init__(self,
                 action_space,
                 cfg=None,
                 std_mode: str = 'fix_no_grad',
                 std_scale=0.1,
                 minimal_std_val=-np.inf,
                 maximal_std_val=np.inf,
                 squash=False, linear=True, nocenter=False):
        super().__init__(action_space)

        self.std_mode = std_mode
        self.std_scale = std_scale  # initial std scale
        self.minimal_std_val = minimal_std_val
        self.maximal_std_val = maximal_std_val
        self.action_scale = action_space.high[0]

        assert std_mode in self.STD_MODES
        n_output = 2 if std_mode == 'statewise' else 1

        self.net_output_dim = self.action_dim * n_output

        if std_mode.startswith('fix'):
            self.log_std = nn.Parameter(torch.zeros(
                1, self.action_dim), requires_grad=('learnable' in std_mode))
        else:
            self.log_std = None

    def get_input_dim(self):
        return self.net_output_dim

    def forward(self, means):
        if self.std_mode.startswith('fix'):  # fix, determine
            log_stds = self.log_std.expand_as(means)
        else:  # 'tanh',
            means, log_stds = torch.chunk(means, 2, dim=-1)

        # if self._cfg.std_mode != 'statewise':
        #from tools.utils import clamp
        #log_stds = torch.clamp(
            # log_stds, minval=max(self.LOG_STD_MIN, self.minimal_std_val), maxval=self.LOG_STD_MAX)
        from tools.utils import clamp
        # log_stds = clamp(min=self.LOG_STD_MIN, max=self.LOG_STD_MAX)
        log_stds = clamp(
            log_stds, minval=max(self.LOG_STD_MIN, self.minimal_std_val), maxval=min(self.LOG_STD_MAX, self.maximal_std_val))
        action_std = torch.exp(log_stds) * self.std_scale
        # else:
        #     log_stds = torch.tanh(log_stds)
        #     log_stds = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_stds + 1) + np.log(self.std_scale)
        #     action_std = torch.exp(log_stds)
                                                                    
        assert not torch.isnan(means).any()

        if not self._cfg.linear:
            if not self._cfg.squash:
                means = torch.tanh(means)
                tanh = False
            else:
                tanh = True
        else:
            tanh = False  # linear gaussian ..

        if self._cfg.nocenter:
            means = means * 0
        return NormalAction(
            means, action_std,
            tanh=tanh, act_scale=self.action_scale
        )
