from ..hidden import HiddenSpace, Normal, merge_a_into_b, CN, Categorical
from gym.spaces import Box


class Goal(HiddenSpace):
    def __init__(
        self, cfg=None, n=6,
        head = Normal.gdc(linear=True, std_mode='fix_no_grad', std_scale=0.3989),
    use_next_state=True
    ) -> None:
        super().__init__()
        self._dim = n 
        self._space = Box(-1, 1, (n,))
        self.head = Normal(self.space, head).cuda()

    @property
    def dim(self):
        return self._dim

    @property
    def space(self):
        return self._space

    def tokenize(self, z):
        return z

    def get_input_dim(self):
        return self.head.get_input_dim()

    def make_policy_head(self, cfg=None):
        # random gaussian ..
        default = Normal.gdc(linear=True, std_scale=1., std_mode='fix_no_grad', nocenter=True, squash=False)
        return Normal(self.space, cfg=merge_a_into_b(CN(cfg), default))

    def get_mask(self, timestep):
        return (timestep > self.max_step - 2).float()

    def likelihood(self, inp, z, timestep):
        return self.head(inp).log_prob(z) * self.get_mask(timestep)

        
    def callback(self, trainer):
        from ..env_base import TorchEnv
        env: TorchEnv = trainer.env
        assert trainer._cfg.time_embedding > 0
        self.max_step = env.max_time_steps

        
class DiscreteGoal(Categorical):
    def __init__(self, cfg=None, use_next_state=True):
        super().__init__()

    def get_mask(self, timestep):
        return (timestep > self.max_step - 2).float()

    def likelihood(self, inp, z, timestep):
        return self.head(inp).log_prob(z) * self.get_mask(timestep)


    def callback(self, trainer):
        from ..env_base import TorchEnv
        env: TorchEnv = trainer.env
        assert trainer._cfg.time_embedding > 0
        self.max_step = env.max_time_steps