import torch
from tools.config import Configurable, as_builder, merge_a_into_b, CN
from nn.distributions.discrete import Discrete
from nn.distributions.normal import Normal, DistHead
from gym.spaces import Box


@as_builder
class HiddenSpace(Configurable):
    def __init__(self, cfg=None,
                 action_weight=1., noise=0.0, obs_weight=1.,
                 use_next_state=False
                 ) -> None:
        super().__init__(cfg)
        self.head = None

    @property
    def dim(self):
        pass

    def tokenize(self, z):
        pass

    def make_policy_head(self, cfg=None):
        raise NotImplementedError

    def get_input_dim(self):
        # return parameters for identify the hidden variables in maximizing the mutual information
        raise NotImplementedError

    def likelihood(self, inp, z, timestep, **kwargs):
        raise NotImplementedError

    def sample(self, inp, mode='sample'):
        return self.head(inp).sample()

    def reward(self, inp, z, timestep):
        return self.likelihood(inp, z, timestep, is_reward=True)

    @property
    def learn(self):
        return True

    def callback(self, trainer):
        # used to 
        pass

        
class Categorical(HiddenSpace):
    def __init__(
        self, cfg=None, n=1,
        head=Discrete.gdc(epsilon=0.2),
    ) -> None:
        super().__init__()
        self.n = n 
        from gym.spaces import Discrete as D
        self._space = D(self.n)
        self.head = Discrete(self.space, cfg=head)

    @property
    def dim(self):
        return self.n

    @property
    def space(self):
        return self._space

    @property
    def learn(self):
        return self.n > 1
        

    def tokenize(self, z):
        return torch.nn.functional.one_hot(z, self.n).float()

    def make_policy_head(self, cfg=None):
        return Discrete(self.space, cfg=cfg)

    def get_input_dim(self):
        return self.n

    def likelihood(self, inp, z, timestep, is_reward=False, **kwargs):
        prob = self.head(inp)
        if not is_reward:
            from tools.utils import logger
            logger.logkv_mean('info_acc', (prob.logits.argmax(dim=-1) == z).float().mean())
            
        return prob.log_prob(z)


class Gaussian(HiddenSpace):
    def __init__(
        self, cfg=None, n=6,
        head = Normal.gdc(linear=True, std_mode='fix_no_grad', std_scale=0.3989),
        obs_dim=1,
    ) -> None:
        super().__init__()
        self._dim = n
        self._space = Box(-1, 1, (n,))
        self.head = Normal(self.space, head).cuda()
        self.obs_dim = obs_dim

    @property
    def dim(self):
        if self.obs_dim == 1:
            return self._dim 
        else:
            return self._dim * (self.obs_dim + 1)

    @property
    def space(self):
        return self._space

    def tokenize(self, z):
        from .utils import positional_encoding
        return positional_encoding(self.obs_dim, z)

    def get_input_dim(self):
        return self.head.get_input_dim()

    def make_policy_head(self, cfg=None):
        default = Normal.gdc(linear=True, std_scale=1., std_mode='fix_no_grad', nocenter=True, squash=False)
        return Normal(self.space, cfg=merge_a_into_b(CN(cfg), default))

    def likelihood(self, inp, z, timestep, **kwargs):
        return self.head(inp).log_prob(z)

    def sample(self, inp, mode='sample'):
        if mode == 'sample':
            return super().sample(inp, mode)
        else:
            # print(inp.shape)
            # print(self.head(inp).dist.loc.shape)
            # print(super().sample(inp, mode).shape)
            return (self.head(inp).dist.loc, None)

            
from nn.space.mixture import MixtureSpace
from nn.distributions.mixture import Mixture as MixtureOutput
class Mixture(HiddenSpace):

    def __init__(
        self, cfg=None, n=6, n_cont=None,
        head=MixtureOutput.gdc(
            discrete=DistHead.to_build(TYPE='Discrete', epsilon=0.05),
            continuous=DistHead.to_build(TYPE='Normal', linear=True, std_mode='fix_no_grad', std_scale=0.3989),
        )
    ) -> None:
        super().__init__()

        if n_cont is not None:
            self.n_cont = n_cont
        else:
            self.n_cont = self._cfg.n//2
        self.n_discrete = self._cfg.n - self.n_cont
        from gym.spaces import Discrete as D

        self._space = MixtureSpace(D(self.n_discrete), Box(-1, 1, (self.n_cont,)))
        self.head = MixtureOutput(self._space, cfg=head)


    @property
    def space(self):
        return self._space

    @property
    def dim(self):
        return self.n_discrete + self.n_cont

    def tokenize(self, z):
        from tools.utils import myround

        z_discrete = myround(z[..., 0])
        z_continuous = z[..., 1:]
        return torch.cat([torch.nn.functional.one_hot(z_discrete, self.n_discrete).float(), z_continuous * 0.3], dim=-1)

    def get_input_dim(self):
        return self.head.get_input_dim()

    def make_policy_head(self, cfg=None):
        default = MixtureOutput.gdc(
            continuous=DistHead.to_build(TYPE='Normal', linear=True, std_scale=1., std_mode='fix_no_grad', nocenter=True, squash=False),
            discrete=DistHead.to_build(TYPE='Discrete', epsilon=0.05)
        )
        return MixtureOutput(self._space, cfg=merge_a_into_b(CN(cfg), default))

    def likelihood(self, inp, z, timestep, **kwargs):
        return self.head(inp).log_prob(z)

    def sample(self, inp, mode='sample'):
        if mode == 'sample':
            return super().sample(inp, mode)
        else:
            return (self.head(inp).dist.loc, None)