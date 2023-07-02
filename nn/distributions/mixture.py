import torch
from tools.utils import batched_index_select
from .dist_head import DistHead, Network, ActionDistr


#WEIGHT_CONTINUOUS = 0.001
WEIGHT_CONTINUOUS = 0.1


class MixtureAction(ActionDistr):
    def __init__(self, discrete, continuous) -> None:
        self.discrete = discrete
        self.continuous = continuous

    def rsample(self, detach=False):
        discrete, discrete_logp = self.discrete.sample()
        continuous, continuous_logp = self.continuous.rsample(detach)
        # print(continuous.shape)
        continuous = batched_index_select(continuous, discrete.dim(), discrete)[..., 0, :]
        # raise NotImplementedError
        continuous_logp = batched_index_select(continuous_logp, discrete.dim(), discrete)[..., 0]

        action = torch.cat([discrete[..., None].float(), continuous], -1)
        assert discrete_logp.shape == continuous_logp.shape
        return action, discrete_logp + continuous_logp * 0.0

    def entropy(self, tolerate=False):
        if not tolerate:
            raise NotImplementedError
        return self.discrete.entropy() #+ self.continuous.entropy()

    def sample(self):
        return self.rsample(detach=True)

    def expand(self, *args, **kwargs):
        return MixtureAction(
            self.discrete.expand(*args, **kwargs),
            self.continuous.expand(*args, **kwargs)
        )

    def get_parameters(self):
        #return self.dist.loc, self.dist.scale
        return self.discrete.get_parameters() + self.continuous.get_parameters()

    def log_prob(self, action):
        from tools.utils import myround
        discrete = myround(action[..., 0])

        discrete_logits = self.discrete.dist.logits

        continuous = action[..., None, 1:].expand(*([-1] * (action.dim() - 1)), discrete_logits.shape[-1], -1)
        dd  = self.discrete.log_prob(discrete) # * 10


        continuous_probs = self.continuous.log_prob(continuous)
        cc = torch.gather(continuous_probs, discrete.dim(), discrete[..., None])[..., 0] * WEIGHT_CONTINUOUS
        return dd + cc


class Mixture(DistHead):
    def __init__(self, action_space, cfg=None, discrete=None, continuous=None):
        #super().__init__(action_space, cfg)

        Network.__init__(self)
        self.discrete = DistHead.build(action_space.discrete, cfg=cfg['discrete'])
        self.continuous = DistHead.build(action_space.continuous, cfg=cfg['continuous']).cuda() # move to cuda.. hack
        self.ddim = self.discrete.get_input_dim()
        self.cdim = self.continuous.get_input_dim()

    def get_input_dim(self):
        return self.ddim + self.cdim * self.ddim

    def forward(self, features):
        discrete, continuous = features[..., :self.ddim], features[..., self.ddim: ].reshape(*features.shape[:-1], self.ddim, self.cdim)
        discrete = self.discrete(discrete)
        continuous = self.continuous(continuous)
        return MixtureAction(discrete, continuous)