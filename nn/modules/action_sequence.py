from turtle import forward
import numpy as np
import torch
from torch import nn
from .backbone import Backbone

class ActionSequence(Backbone):

    def __init__(
        self,
        obs_space,
        action_space,
        cfg=None,
        a_size=0
    ):
        super().__init__(obs_space, action_space, cfg)
        # import code
        # code.interact(local=locals())
        self.seq = torch.nn.Parameter(
            torch.zeros(
                100, a_size
            )
        )
        self._output_shape = (a_size, )

    def forward(self, x=None, a=None, timestep=None):
        if (timestep is None):
            return NotImplementedError
        return self.seq[timestep]