import torch
from gym.spaces import Space, Box
from torch import nn
from tools import Configurable

from tools.config import CN
from .head import Head
from .backbone import Backbone
from .head import Linear
from .backbone import MLP

from tools.nn_base import Network as NetworkBase

class MySequential(torch.nn.Sequential):
    def forward(self, *input):
        for module in self:
            input = [module(*input)]
        return input[0]


class Network(NetworkBase):
    def __init__(
            self,
            cfg: CN = None,
            backbone: CN = Backbone.to_build(TYPE=MLP),
    ):
        super().__init__()
        self.backbone = backbone


class Actor(Network):
    def __init__(self,
                 obs_space: Space,
                 action_space: Box,
                 auxiliary: int,
                 cfg: CN = None,
                 head=Head.to_build(TYPE='PPOHead')
                 ):
        super(Actor, self).__init__(cfg=cfg)
        self.backbone = Backbone.build(obs_space, None, cfg=self._cfg.backbone)
        self.head = Head.build(self.backbone.output_shape, action_space, cfg=head)
        if auxiliary:
            self.aux = Linear(self.backbone.output_shape, auxiliary)
        else:
            self.aux = None

    def forward(self, state: torch.Tensor, aux=False, timestep=0):
        feat = self.backbone(state)
        action = self.head(feat)
        if aux:
            return action, self.aux(feat)
        else:
            return action


class Value(Network):
    def __init__(self, obs_space: Box, cfg: CN = None, output_dim=1):
        Network.__init__(self, cfg=cfg)
        self.backbone = Backbone.build(obs_space, None, cfg=self.backbone)
        self.head = Head.build(self.backbone.output_shape, output_dim, TYPE="Linear")

    def forward(self, x: torch.Tensor):
        return self.head(self.backbone(x))


class DoubleCritic(Network):
    def __init__(self, obs_space: Box, action_space, cfg: CN = None, output_dim=1):
        Network.__init__(self, cfg=cfg)
        self.backbone1 = Backbone.build(
            obs_space,
            action_space,
            cfg=self.backbone
        )
        self.backbone2 = Backbone.build(
            obs_space,
            action_space,
            cfg=self.backbone
        )
        self.head1 = Head.build(
            self.backbone1.output_shape,
            output_dim,
            TYPE="Linear"
        )
        self.head2 = Head.build(
            self.backbone2.output_shape,
            output_dim,
            TYPE="Linear"
        )

    def forward(self, x: torch.Tensor, a: torch.Tensor, q1=False):
        Q1 = self.head1(self.backbone1(x, a))
        if q1:
            return Q1
        Q2 = self.head2(self.backbone2(x, a))
        return Q1, Q2
