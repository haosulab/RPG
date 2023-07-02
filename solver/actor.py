# Actor: Backbone + a Distribution Head
#   map obs_space to action space through a distribution defined by DistHead
#   Two kinds of backbone:  
#     1. a neural network defined by a Backbone; note that the backbone can take a tuple as the observation for conditioning on the latent variables.
#     2. or parameter sequences

import torch
from torch import nn
from tools.nn_base import Network, intprod
from gym.spaces import Discrete


from nn.modules import Backbone, MLP
from nn.distributions import Normal, DistHead


class Actor(Network):
    # a network module that maps an observation to an action 
    # support different kinds of backbones and distribution heads
    # the obs_space can be either a space, or a tuple of (obs_space, hidden_space)

    def __init__(
        self, obs_space, action_space, cfg=None,
        backbone=Backbone.to_build(TYPE=MLP),
        head=DistHead.to_build(TYPE=Normal),
        timestep=0, multihead=False,
    ):
        super().__init__()

        self.obs_space = obs_space
        self.action_space = action_space

        self.head = DistHead.build(action_space, cfg=head)

        if isinstance(obs_space, tuple) and isinstance(obs_space[1], Discrete):
            # take special care of the discrete input .. 
            self.cond_z = obs_space[1].n
            if multihead:
                assert len(obs_space) == 2
                obs_space = obs_space[0]
        else:
            self.cond_z = 0

        if timestep > 0:
            # Actor can also support non-neural network mode ..
            if self.cond_z > 0:
                timestep = timestep * self.cond_z
            # break symmetry..
            self.param = nn.Parameter(torch.randn(
                (timestep, self.head.get_input_dim()), dtype=torch.float32) * 0.0001)

        else:
            self.backbone = Backbone.build(obs_space, cfg=backbone)
            input_dim = intprod(self.backbone.output_shape)

            param_size = self.head.get_input_dim()
            if self.cond_z > 0 and multihead:
                param_size = param_size * self.cond_z

            #if (hasattr(head, "linear") and head.linear) or (not hasattr(head, "linear")):
            self.linear = nn.Linear(input_dim, param_size)
            self.linear.weight.data.mul_(0.1)
            self.linear.bias.data.mul_(0.0)


    def forward(self, state, *, timestep=None):
        if self._cfg.timestep > 0:
            if self.cond_z:
                if self.cond_z > 0:
                    assert len(state) == 2, "state should be a tuple of (obs, z) and z only has one dimension"
                    state, z = state

                index = timestep * self.cond_z + z
                feat = self.param[index]
            else:
                feat = self.param[timestep]
                feat = feat[None, :].expand(len(state), *feat.shape)
        else:
            if self.cond_z > 0  and self._cfg.multihead:
                state, z = state

            feat = self.backbone(state)
            #if hasattr(self, "linear"):
            feat = self.linear(feat)
            if self.cond_z > 0 and self._cfg.multihead:
                assert z is not None
                assert z.dim() == 1
                feat = feat.reshape(*feat.shape[:-1], -1, self.cond_z)
                feat = feat[torch.arange(len(z), device=self.device), ..., z]
                # cc = feat
                # assert torch.allclose(cc[2, ..., z[2]], feat[2])
                # print(feat.shape)
                raise NotImplementedError("multihead not implemented")

        return self.head(feat)


    # def render(self, print_fn):
    #     if print_fn is None: print_fn = print
    #     # code for visualizing the parameters
    #     if self._cfg.timestep > 0:
    #         z_dims = [0] if not self.cond_z else list(range(self.cond_z))
    #         for j in range(min(self._cfg.timestep, 2)):
    #             for i in z_dims[:3]:
    #                 dist = self.forward(([0], torch.LongTensor([i]).to(self.device)), timestep=j)
    #                 dist.render(lambda *args: print_fn(f"mode {i}" if self.cond_z else '', *args))
    #     else:
    #         raise NotImplementedError