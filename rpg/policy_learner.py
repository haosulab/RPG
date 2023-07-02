import torch
from nn.space import Discrete
from gym.spaces import Box
from tools.optim import LossOptimizer
from tools.config import Configurable
from tools.utils import logger
from tools.utils.scheduler import Scheduler
from tools.config import Configurable, as_builder
from collections import namedtuple
from nn.distributions import Normal


class EntropyLearner(Configurable):
    def __init__(
        self,
        name, space,
        cfg=None,
        coef=1.,
        target_mode='auto',
        target=None,
        lr=3e-4,
        device='cuda:0',
        schedule=Scheduler.to_build(
            TYPE='constant'
        )
    ):
        super().__init__(cfg)
        self.name = name

        if target is not None or target_mode == 'auto':
            if target == None:
                if isinstance(space, Box):
                    target = -space.shape[0]
                elif isinstance(space, Discrete):
                    target = space.n
                else:
                    raise NotImplementedError
            self.target  = target
            self.log_alpha = torch.nn.Parameter(torch.zeros(1, device=device))
            self.optim = LossOptimizer(self.log_alpha, lr=lr) #TODO: change the optim ..
        else:
            self.target = None
            self.schedule: Scheduler = Scheduler.build(cfg=schedule)

        
    def update(self, entropy):
        if self.target is not None:
            entropy_loss = -torch.mean(self.log_alpha.exp() * (self.target - entropy.detach()))
            self.optim.optimize(entropy_loss)
        else:
            self.schedule.step()

        logger.logkv_mean(self.name + '_alpha', float(self.alpha))
        logger.logkv_mean(self.name + '_ent', float(entropy.mean()))

    @property
    def alpha(self):
        if self.target is None:
            return self.schedule.get() * self._cfg.coef
        return float(self.log_alpha.exp() * self._cfg.coef)
        

Zout = namedtuple('Zout', ['a', 'logp', 'entropy', 'new', 'logp_new'])

def select_newz(policy, state, hidden, alpha, z, timestep, K):
    new = (timestep % K == 0)
    log_new_prob = torch.zeros_like(new).float()
    z = z.clone()

    logp_z = torch.zeros_like(log_new_prob)
    entropy = torch.zeros_like(log_new_prob) 

    if new.any():
        newz, newz_logp, ent = policy(state[new], hidden[new] if hidden is not None else None, alpha)
        z[new] = newz
        logp_z[new] = newz_logp
        entropy[new] = ent
    return Zout(z, logp_z, entropy, new, log_new_prob)


class PolicyLearner(LossOptimizer):
    def __init__(self, name, action_space, policy, enc_z, cfg=None,
                 ent=EntropyLearner.dc, freq=1000000, max_grad_norm=1., lr=3e-4, ignore_hidden=False):
        from tools.utils import orthogonal_init
        policy.apply(orthogonal_init)

        super().__init__(policy, cfg)
        self.name = name
        self._policy = policy
        self.enc_z = enc_z
        self.ent = EntropyLearner(name, action_space, cfg=ent)
        self.freq = freq
        import copy
        self._target_policy = copy.deepcopy(self._policy)
        self.mode = 'train'

    def update(self, rollout):
        ent = rollout[f'ent_{self.name}']
        loss = self.policy.loss(rollout, self.ent.alpha)
        logger.logkv_mean(self.name + '_loss', float(loss))
        self.optimize(loss)
        self.ent.update(ent)

    @property
    def policy(self):
        return self._policy if self.mode == 'train' else self._target_policy

    def set_mode(self, mode):
        self.mode = mode

    def ema(self, decay=0.999):
        from tools.utils import ema
        ema(self._policy, self._target_policy, decay)

    def __call__(self, s, hidden, prev_action=None, timestep=None):
        inps = [s]
        hidden = self.enc_z(hidden) if not self._cfg.ignore_hidden else None
        #s = self.policy.add_alpha(*inps, timestep=timestep) # concatenate the two
        

        with torch.no_grad():
            alpha = self.ent.alpha

        if prev_action is not None:
            assert timestep is not None
            # TODO: allow to ignore the previous? not need now
            return select_newz(self.policy, s, hidden, alpha, prev_action, timestep, self.freq)
        else:
            return self.policy(s, hidden, alpha)

    def intrinsic_reward(self, rollout):
        with torch.no_grad():
            alpha = self.ent.alpha
        name = 'ent_{}'.format(self.name)
        if name not in rollout:
            return None, None
        return name, rollout[name] * alpha