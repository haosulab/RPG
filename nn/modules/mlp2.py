# base network..
import numpy as np
import torch
from torch import nn
from .backbone import Backbone
from tools.nn_base import Network
from gym.spaces import Discrete, Box


ACTIVATIONS = {
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
}


class MLPBase(Network):
    def __init__(
        self,
        obs_space,
        action_space,
        cfg=None,

        dims=(256, 256),
        layers=None,
        dim_per_layer=256,
        output_relu=True,
        activation='relu'
    ):
        #super().__init__(obs_space, action_space, cfg=cfg)
        super().__init__()
        self.obs_space = obs_space
        self.action_space = action_space

        if layers is not None:
            dims = (dim_per_layer,) * layers

        input_shape = list(obs_space.shape)
        assert len(obs_space.shape) == 1
        self.x_dim = obs_space.shape[0]

        if action_space is not None:
            input_shape[0] += action_space.shape[0]
            assert len(action_space.shape) == 1
            self.action_dim = action_space.shape[0]

        dims = (int(np.prod(input_shape)),) + dims

        self.dims = dims
        self.input_shape = input_shape
        assert len(dims) >= 2

        layers = []

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            if i != len(dims) - 2 or output_relu:
                layers.append(ACTIVATIONS[activation]())

        self.main = nn.Sequential(*layers)
        self._output_shape = (dims[-1],)

    def forward(self, x, a=None, *, timestep=None):
        x = self.batch_input(x)
        x_shape = x.shape
        x = x.reshape(*x_shape[:-1], self.x_dim)

        if a is not None:
            a = self.batch_input(a)
            a = a.reshape(*a.shape[:-1], self.action_dim)
            x = torch.cat((x, a), -1)
            assert self.action_space is not None

        assert x.shape[-1] == self.dims[0], f"{x.shape[1], self.dims[0]}"
        output = self.main(x)
        return output



class MLP(Backbone):
    # MLP that supports tuple of observations ..
    def __init__(self, obs_space, cfg=None):
        super().__init__(obs_space)
        self.mlp = MLPBase(self.obs_space, self.action_space)
        self._output_shape = self.mlp._output_shape

    def forward(self, x, *, timestep=None):
        x, a = self.preprocess(x, timestep=timestep)
        return self.mlp(x, a, timestep=timestep)
