import torch
from tools import as_builder
from tools.nn_base import Network, intprod


class ActionDistr:
    def sample(self, *args, sum=True, **kwargs):
        raise NotImplementedError

    def rsample(self, *args, sum=True, **kwargs):
        raise NotImplementedError

    def log_prob(self, action, sum=True):
        raise NotImplementedError

    def get_parameters(self):
        raise NotImplementedError


    def __iter__(self):
        for a, b in zip(*self.get_parameters()):
            yield self.__class__(a, b)

    def stack(self, args, axis=0):
        raise NotImplementedError
        return self.__class__(*[torch.stack(x, axis=axis) for x in zip(*[i.get_parameters() for i in args])], **self.kwargs)

    def concat(self, args, axis=0):
        raise NotImplementedError
        return self.__class__(*[torch.concat(x, axis=axis) for x in zip(*[i.get_parameters() for i in args])], **self.kwargs)

    def to(self, device):
        return self.__class__(*[i.to(device) for i in self.get_parameters()], **self.kwargs)

    def expand(self, batch_size):
        assert isinstance(batch_size, int)
        return self.__class__(*[i[None, :].expand(batch_size, *i.shape) for i in self.get_parameters()], **self.kwargs)

    def render(self, print_fn):
        raise NotImplementedError


@as_builder
class DistHead(Network):
    # turn network output into an action_distr

    def __init__(self, action_space, cfg=None):
        low, high = action_space.low, action_space.high
        assert (low == low[0]).all() and (high == high[0]).all(), f"{low} {high}"
        self.action_dim = low.shape[0]
        super().__init__()

    def get_input_dim(self):
        raise NotImplementedError

    def forward(self, features):
        raise NotImplementedError
