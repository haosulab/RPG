from torch.distributions import Uniform
from rl.models.action import ActionDistr

class UniformAction(ActionDistr):
    PYTORCH_DISTRIBUTION = Uniform

    def get_parameters(self):
        return self.dist.low, self.dist.high

