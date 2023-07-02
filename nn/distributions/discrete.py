import torch
from .dist_head import ActionDistr
from torch.distributions import Categorical
from .dist_head import DistHead, Network
from gym.spaces import Discrete


class CategoricalAction(ActionDistr):
    # with epsilon, do random sample ..
    def __init__(self, logits, epsilon=0.):
        self.logits = logits
        self.kwargs = {'epsilon': epsilon}
        self.epsilon = epsilon
        if self.epsilon > 0:
            logits = torch.log(torch.softmax(logits, -1) * (1-self.epsilon) + self.epsilon/logits.shape[-1])
            logits = torch.log_softmax(logits, -1)
            # raise NotImplementedError
        self.dist = Categorical(logits=logits)

    def rsample(self, *args, **kwargs):
        raise NotImplementedError(
            "Do not allow sample from a categorical distirbution for optimization ..")

    def log_prob(self, action):
        return self.dist.log_prob(action)

    def entropy(self):
        return self.dist.entropy()

    def sample(self):
        action = self.dist.sample()
        return action, self.dist.log_prob(action)

    def get_parameters(self):
        return (self.logits,)

    def render(self, print_fn):
        print_fn(torch.softmax(self.logits, -1).detach().cpu().numpy())


class Discrete(DistHead):
    def __init__(self, action_space: Discrete, cfg=None, epsilon=0.):
        Network.__init__(self)
        self.action_dim = action_space.n

    def get_input_dim(self):
        return self.action_dim

    def forward(self, inp):
        return CategoricalAction(inp, epsilon=self._cfg.epsilon)
