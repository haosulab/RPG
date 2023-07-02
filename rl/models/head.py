import numpy as np
import torch
from typing import Optional, Union

import numpy as np
import torch
from torch import nn

from .utils import intprod
from typing import Tuple, Optional, Union
from tools.config import CN, Configurable, as_builder
from .action import NormalAction
from gym.spaces import Space
from tools.nn_base import Network


@as_builder
class Head(Network):
    residual = False
    def __init__(self, feature_shape: Tuple[int],
                 output_space: Union[Space, int],
                 cfg: Optional[CN] = None):
        super().__init__()
        self.output_space = output_space


class Linear(Head):
    def __init__(self, feature_shape: Tuple[int],
                       output_space: int,
                       cfg: CN = None):
        assert isinstance(output_space, int)
        super(Linear, self).__init__(feature_shape, cfg)
        input_dim = intprod(feature_shape)
        self.linear = nn.Linear(input_dim, output_space)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, dict): x = x['feature']
        out = self.linear(x)
        return out

from torch.autograd import Function
class DiffClamp(Function):

    @staticmethod
    def forward(ctx, i, min_val, max_val):
        # ctx._mask = (i.ge(min_val) * i.le(max_val))
        return i.clamp(min_val, max_val)

    @staticmethod
    def backward(ctx, grad_output):
        #mask = Variable(ctx._mask.type_as(grad_output.data))
        return grad_output, None, None



class PPOHead(Linear):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -20

    STD_MODES = ['statewise', 'fix_learnable', 'fix_no_grad']

    def __init__(self,
                 feature_shape: Tuple[int],
                 output_space: Space,
                 cfg: CN=None,
                 std_mode: str='fix_learnable',
                 std_scale=float(np.exp(-0.5)),
                 minimal_std_val=-np.inf,
                 squash=False,
                 use_gmm=0,
                 ):

        self.std_mode = std_mode
        self.std_scale = std_scale # initial std scale
        self.minimal_std_val = minimal_std_val

        assert std_mode in self.STD_MODES
        n_output = 2 if std_mode == 'statewise' else 1
        self.output_dim = intprod(output_space.shape)

        net_output_dim = self.output_dim * n_output
        if use_gmm:
            net_output_dim = (n_output * self.output_dim + 1) * use_gmm 

        super(PPOHead, self).__init__(
            feature_shape,
            net_output_dim
        )

        self.output_space = output_space

        self.linear.weight.data.mul_(0.1)
        self.linear.bias.data.mul_(0.0)

        low, high = output_space.low, output_space.high

        self.action_scale = nn.Parameter(
            torch.tensor((high - low) / 2),
            requires_grad=False)

        self.action_bias = nn.Parameter(
            torch.tensor((high + low) / 2),
            requires_grad=False)

        if std_mode.startswith('fix'):
            self.log_std = nn.Parameter(
                torch.zeros(1, self.output_dim),
                requires_grad=('learnable' in std_mode),
            )
        else:
            self.log_std = None

        self.clamp = DiffClamp.apply


    def forward(self, x):
        means = self.linear(x)
        if self._cfg.use_gmm > 0:
            log_loc = means[:, :self._cfg.use_gmm]
            means = means[:, self._cfg.use_gmm:]
            means = means.reshape(means.shape[0], self._cfg.use_gmm, -1)

        if self.std_mode.startswith('fix'):
            # fix, determine
            log_stds = self.log_std.expand_as(means)
        else:
            # 'tanh', 
            means, log_stds = torch.chunk(means, 2, dim=-1)

        log_stds = torch.clamp(
            log_stds, min=max(self.LOG_STD_MIN, self.minimal_std_val), max=self.LOG_STD_MAX)

        action_std = torch.exp(log_stds) * self.std_scale
        assert not torch.isnan(means).any()

        if self._cfg.use_gmm == 0:
            if not self._cfg.squash:
                center = torch.tanh(means.double())
                return NormalAction(
                    center.float() * self.action_scale + self.action_bias,
                    action_std * self.action_scale.clamp(1e-15, np.inf)
                )
            else:
                assert self.action_scale.mean() == 1. and self.action_bias.mean() == 0.
                return NormalAction(means, action_std, tanh=True)
        else:
            means = torch.tanh(means)
            from solver.mixture_of_guassian import GMMAction
            return GMMAction(log_loc,
                             means * self.action_scale + self.action_bias,
                             action_std * self.action_scale.clamp(1e-15, np.inf)
                             )
