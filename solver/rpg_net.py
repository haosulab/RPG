import torch
from gym.spaces import Box, Discrete
from tools.nn_base import Network, intprod
from tools.utils import batch_input
from tools.config import as_builder

from .utils import create_prior
from .actor import Actor, DistHead
from nn.distributions import DeterminisiticAction

def select_new_z(timestep, K):
    return timestep == 0 or (K and timestep % K == 0)


def make_zhead(z_space, continuous, discrete):
    if isinstance(z_space, Discrete):
        z_head = discrete
    elif isinstance(z_space, Box):
        z_head = continuous
    else:
        z_head = dict(
            TYPE="Mixture",
            discrete=discrete,
            continuous=continuous,
        )
    return z_head

@as_builder
class RPGActor(Network):
    def __init__(
        self, obs_space, action_space, z_space, K, 
        cfg=None, # K
        not_func=False,
        a_head=DistHead.to_build(
            linear=True,
            squash=False,
            std_mode='fix_no_grad',
            std_scale=0.01
        ), # by default we use a near determinstic policy 
        backbone=None, env_low_steps=None,
        multihead=False,
        ignore_previous_z=False,
        
        softmax_policy = False,
    ):
        super().__init__()

        discrete = dict(TYPE='Discrete', epsilon=0.0)  # 0.2 epsilon
        continuous = dict(TYPE='Normal', linear=True, std_mode='fix_no_grad', std_scale=1., nocenter=True)
        z_head = make_zhead(z_space, continuous, discrete)


        extra_kwargs = {}
        if not_func:
            assert env_low_steps is not None
            extra_kwargs['timestep'] = env_low_steps

        self._pi_a = Actor(
            (obs_space, z_space),
            action_space,
            backbone=backbone,
            head=a_head,
            **extra_kwargs,
            multihead=multihead
        )
        self._pi_z = Actor(
            (obs_space, z_space),
            z_space,
            backbone=backbone,
            head=z_head,
            **extra_kwargs,
            multihead=multihead
        )
        self.K = K
        if self._cfg.softmax_policy: 
            import numpy as np
            self.softmax_temperature = 1.
            n = z_space.n #must for detemrinsitic env
            assert self.K == 0
            self.counter = np.array([1. for i in range(n)])
            self.rewards = np.array([0. for i in range(n)])
    
    def pi_a(self, obs, z, *, timestep=None):
        return self._pi_a((obs, z), timestep=timestep)

    def pi_z(self, obs, z, *, timestep=None):
        assert timestep is not None
        if select_new_z(timestep, self.K):
            if self._cfg.ignore_previous_z:
                z = z * 0
            if self._cfg.softmax_policy:
                assert timestep == 0
                from .distributions.discrete import CategoricalAction
                import numpy as np
                from tools.utils import totensor
                logits = totensor(np.array(self.rewards)/np.array(self.counter), device='cuda:0')
                assert z.dim() == 1
                logits = logits[None, :].expand(z.shape[0], -1)
                if self.softmax_temperature > 0.:
                    logits = logits / self.softmax_temperature
                else:
                    #logits = logits * 0.
                    p = torch.zeros_like(logits)
                    p[:, logits[0].argmax()] = 1e9
                    logits = p
                return CategoricalAction(logits)
            else:
                p_z = self._pi_z((obs, z), timestep=timestep)
        else:
            p_z = DeterminisiticAction(z)
        return p_z

    # def render(self, print_fn):
    #     print_fn('z: ')
    #     self._pi_z.render(print_fn)
    #     print_fn('a: ')
    #     self._pi_a.render(print_fn)



class Identity(Network):
    def __init__(self, x, cfg=None):
        Network.__init__(self)
        self.x = x
    
    def forward(self, obs, *args, **kwargs):
        if isinstance(obs, tuple): obs = obs[0]
        return self.x.expand(len(obs))

class PriorActor(RPGActor):
    def __init__(self, obs_space, action_space, z_space, K, cfg=None):
        Network.__init__(self)
        self._pi_a = Identity(create_prior(action_space))
        self._pi_z = Identity(create_prior(z_space))
        self.K = K
        self.action_space = action_space
        self.z_space = z_space

    def to(self, device):
        self._pi_a = Identity(create_prior(self.action_space, device=device))
        self._pi_z = Identity(create_prior(self.z_space, device=device))
        return super().to(device)


@as_builder
class RPGCritic(Network):
    def __init__(self, obs_space, z_space, K, cfg=None, backbone=None):
        super().__init__()
        # V(s, z) where z is already sampled .. critic for a
        self.main = Actor((obs_space, z_space), Box(-1., 1., (1,)), head=dict(TYPE='Deterministic'), backbone=backbone) 

        # V(s, z) before selecting the next z.. critic for z
        self.main2 = Actor(obs_space, Box(-1., 1., (1,)), head=dict(TYPE='Deterministic'), backbone=backbone) 
        self.K = K

    def forward(self, obs, z, z_old, *, timestep=None):
        v_a = self.main((obs, z))
        if select_new_z(timestep, self.K):
            v_z = self.main2(obs)
        else:
            v_z = v_a
        return torch.stack((v_z, v_a), -1)


@as_builder
class InfoNet(Network):
    # should use a transformer instead ..
    def __init__(self, obs_space, action_space, z_space, cfg=None, backbone=None, action_weight=1., noise=0.0, obs_weight=1.):
        super().__init__()
        discrete = dict(TYPE='Discrete', epsilon=0.0)  # 0.2 epsilon
        continuous = dict(TYPE='Normal', linear=True, std_mode='fix_no_grad', std_scale=1.)
        z_head = make_zhead(z_space, continuous, discrete)
        self.main = Actor((obs_space, action_space), z_space,
                          backbone=backbone, head=z_head).cuda()
        self.preprocess = None

    def forward(self, s, a, z, *, timestep=None):
        #if timestep is None:
        #    timestep = torch.arange(len(s), device=self.device)
        from tools.utils import dmul, dshape

        if self.preprocess is not None:
            s = self.preprocess(s)

        return self.main(
            (dmul(s, self._cfg.obs_weight),
             (a + torch.randn_like(a) * self._cfg.noise) * self._cfg.action_weight)
        ).log_prob(z)
