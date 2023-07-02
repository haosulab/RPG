import torch
from .dist_head import DistHead, ActionDistr


class DeterminisiticAction(ActionDistr):
    def __init__(self, action, allow_not_equal=False):
        self.action = action
        self.kwargs = {}
        self.allow_not_equal = allow_not_equal

    def sample(self):
        return self.action.detach(), self.log_prob(self.action)

    def rsample(self):
        return self.action, self.log_prob(self.action)

    def log_prob(self, z):
        #raise NotImplementedError
        # print(self.action.dtype)
        if not self.allow_not_equal:
            assert torch.allclose(z, self.action), f"{z} != {self.action}"
        return torch.zeros(len(self.action), device=self.action.device, dtype=torch.float32)

    def entropy(self):
        return torch.zeros(len(self.action), device=self.action[0].device, dtype=torch.float32)

    def get_parameters(self):
        return self.action,

    @property
    def mean(self):
        return self.action

    @property
    def scale(self):
        return self.action * 0


class Deterministic(DistHead):
    def __init__(self, action_space, cfg=None):
        super().__init__(action_space)
        self.net_output_dim = self.action_dim

    def get_input_dim(self):
        return self.net_output_dim

    def forward(self, means):
        return DeterminisiticAction(means)
