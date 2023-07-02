import torch
import numpy as np
from torch import nn
from tools.config import Configurable, as_builder
from tools.utils import batch_input
from tools.nn_base import Network
from gym.spaces import Discrete, Box, Space
from nn.space import MixtureSpace



@as_builder
class Backbone(Network):
    # the backbone will support tuple of observation spaces ..
    # in this case it accepts either network(obs, z) or network((obs, z)). The later is prefered.

    def __init__(
        self,
        obs_space: Space,
        cfg=None
    ):
        Network.__init__(self)
        self._original_obs_space = obs_space

        if isinstance(obs_space, tuple):
            assert len(obs_space) == 2
            obs_space, action_space = obs_space[0], obs_space[1]
        else:
            action_space = None

        self.old_action_space = action_space
        if action_space is not None:
            if isinstance(action_space, Discrete):
                 action_space = Box(-1, 1, (action_space.n,))
            elif isinstance(action_space, Box):
                pass
            elif isinstance(action_space, MixtureSpace):
                action_space = Box(-1, 1, (action_space.discrete.n + action_space.continuous.shape[-1],))
            else:
                raise NotImplementedError("action space not supported {}".format(action_space))

        self.obs_space = obs_space
        self.action_space = action_space
    
    @property
    def output_shape(self):
        return self._output_shape

    def preprocess(self, x, *, timestep=None):
        # a strong type check..
        assert isinstance(x, tuple) == isinstance(
            self._original_obs_space, tuple), f"{x}, {self._original_obs_space}"

        if isinstance(x, tuple):
            #x, a = x
            assert len(x) == 2
            x, a = x[0], x[1]
        else:
            a = None

        def onehot(a, space):
            return torch.nn.functional.one_hot(self.batch_input(a), num_classes=space.n)
        from tools.utils import myround


        space = self.old_action_space
        if space is not None:
            if isinstance(space, Discrete):
                a = onehot(a, space)
            elif isinstance(space, Box):
                pass
            else:
                a = torch.concat([onehot(myround(a[..., 0]), space.discrete), self.batch_input(a[..., 1:])], dim=-1)
                assert a.shape[-1] == self.action_space.shape[-1], f"{a.shape}, {self.action_space.shape[-1]}"

        return x, a