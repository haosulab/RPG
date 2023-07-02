import torch
from torch.distributions import Normal
from .backbone import batch_input


class ActionDistr:
    PYTORCH_DISTRIBUTION = None

    def __init__(self, *args, tanh=False, **kwargs):
        self.dist = self.PYTORCH_DISTRIBUTION(*args, **kwargs)
        self.tanh = tanh
        self.kwargs = {'tanh': tanh}

    def sample(self, *args, sum=True, **kwargs):
        action = self.dist.sample(*args, **kwargs)
        logp = self.dist.log_prob(action.detach()) # detach here ..
        if sum is True:
            logp = logp.sum(dim=-1)
        if self.tanh:
            action = torch.tanh(action)
            logp -= torch.log(1. * (1 - action.pow(2)) + 1e-6).sum(axis=-1)
        return action, logp

    def rsample(self, *args, sum=True, **kwargs):
        action = self.dist.rsample(*args, **kwargs)
        #self.dist.loc.register_hook(print)
        logp = self.dist.log_prob(action) # do not detach
        #logp.register_hook(print)
        if sum is True:
            logp = logp.sum(dim=-1)
        if self.tanh:
            action = torch.tanh(action)
            logp -= torch.log(1. * (1 - action.pow(2)) + 1e-6).sum(axis=-1)
        return action, logp

    def batch_action(self, action):
        return batch_input(action, self.get_parameters()[0].device)

    def log_prob(self, action, sum=True):
        action = self.batch_action(action)
        log_probs = self.dist.log_prob(action.detach())
        if sum:
            log_probs = log_probs.sum(dim=-1)
        return log_probs

    def entropy(self, sum=True):
        entropy = self.dist.entropy()
        if sum:
            entropy = entropy.sum(dim=-1)
        return entropy

    @property
    def mean(self):
        raise NotImplementedError

    def get_parameters(self):
        raise NotImplementedError

    def __iter__(self):
        for a, b in zip(*self.get_parameters()):
            yield self.__class__(a, b)

    @classmethod
    def stack(cls, args, axis=0):
        return cls(*[torch.stack(x, axis=axis) for x in zip(*[i.get_parameters() for i in args])])

    @classmethod
    def concat(cls, args, axis=0):
        return cls(*[torch.concat(x, axis=axis) for x in zip(*[i.get_parameters() for i in args])])

    def to(self, device):
        return self.__class__(*[i.to(device) for i in self.get_parameters()])


    def REINFORCE(self, enforce=False):
        # compute the gradient direclty
        raise NotImplementedError('do not support REINFORCE by default')

    def expand(self, batch_size):
        assert isinstance(batch_size, int)
        return self.__class__(*[i[None, :].expand(batch_size, *i.shape) for i in self.get_parameters()], **self.kwargs)

class NormalAction(ActionDistr):
    PYTORCH_DISTRIBUTION = Normal

    @property
    def mean(self):
        return self.dist.loc

    @property
    def loc(self):
        return self.dist.loc

    @property
    def scale(self):
        return self.dist.scale

    @property
    def batch_size(self):
        return self.mean.shape[0]

    def get_parameters(self):
        return self.loc, self.scale
    
    def REINFORCE(self, enforce=False):
        # if we use rsample, there is not need for reinforce
        return 0.