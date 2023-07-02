import torch
from tools.utils import logger
from tools.nn_base import Network
from nn.distributions import DistHead, NormalAction
from tools.utils import Seq, mlp
from tools.optim import LossOptimizer
from tools.config import Configurable
from tools.utils import totensor
from nn.space import Discrete
from gym.spaces import Box
from einops import repeat

def seg2rollout(seg):
    if not isinstance(seg, dict):
        obs = seg.obs_seq
        a = seg.action
        z = seg.z

    rollout = {
        'state': obs,#torch.stack([obs, obs], dim=0),
        'a': a,
        'z': repeat(z, '... -> b ...', b=a.shape[0]),
        'timestep': seg.timesteps,
    }
    return rollout


class InfoNet(Network):
    def __init__(self, 
                state_dim, action_space, hidden_dim, hidden_space, cfg=None, learn_posterior=False):
        super().__init__()
        action_dim = action_space.shape[0]
        from .hidden import HiddenSpace
        self.hidden: HiddenSpace = hidden_space
        self.config = self.hidden._cfg

        self.info_net = Seq(mlp(state_dim + (action_dim if self.config.action_weight > 0. else 0),
                                hidden_dim, self.hidden.get_input_dim()))


        # if learn_posterior:
        #     self.posterior_z = Seq(mlp(state_dim, hidden_dim, self.hidden.get_input_dim()))

    def compute_feature(self, states, a_seq,):
        states = states * self.config.obs_weight
        if self.config.action_weight > 0.:
            a_seq = (a_seq + torch.randn_like(a_seq) * self.config.noise)
            a_seq = a_seq * self.config.action_weight
            return self.info_net(states, a_seq)
        else:
            return self.info_net(states)

    def get_state_seq(self, traj):
        if self.config.use_next_state:
            return traj['state'][1:]
        return traj['state'][:-1]

    def forward(self, traj, mode='likelihood'):
        states = self.get_state_seq(traj)
        a_seq = traj['a']
        z_seq = traj['z']

        if mode != 'reward':
            states = states.detach()
            a_seq = a_seq.detach()
            z_seq = z_seq.detach()

        inp = self.compute_feature(states, a_seq)
        if mode != 'sample':
            t = traj['timestep']
            if mode == 'likelihood':
                return self.hidden.likelihood(inp, z_seq, timestep=t)[..., None]
            elif mode == 'reward':
                return self.hidden.reward(inp, z_seq, timestep=t)[..., None]
        else:
            return self.hidden.sample(inp)

    def enc_s(self, obs, timestep):
        return self.enc_s(obs, timestep=timestep)

    def get_posterior(self, state, z=None):
        #return self.posterior_z(states)
        inp = self.posterior_z(state)
        if z is not None:
            return self.hidden.likelihood(inp, z, timestep=None)
        else:
            return self.hidden.sample(inp, mode='mean')


from tools.utils.scheduler import Scheduler
class InfoLearner(LossOptimizer):
    def __init__(self, obs_space, state_dim, action_space, hidden_space, cfg=None,
                 net=InfoNet.dc,
                 coef=1.,
                 weight=Scheduler.to_build(TYPE='constant'),
                 hidden_dim=256, #learn_posterior=False,

                 use_latent=True,  # if this is not True, the network will take the original observation as input
                 learn_with_dynamics=False, # if this is true, the network will be trained with dynamics model
        ):
        assert use_latent
        self.use_latent = use_latent
        self.learn_with_dynamics = learn_with_dynamics

        if learn_with_dynamics:
            assert use_latent, 'learn_with_dynamics requires use_latent'
            # assert not learn_posterior, 'learn_with_dynamics requires not learn_posterior'

        if not use_latent:
            state_dim = obs_space.shape[0]

        net = InfoNet(
            state_dim, action_space, hidden_dim, hidden_space, cfg=net,
            # learn_posterior=learn_posterior
        ).cuda()
        self.coef = coef
        self.info_decay: Scheduler = Scheduler.build(weight)
        # self.learn_posterior = learn_posterior
        super().__init__(net)
        self.net = net
        import copy
        self.target_net = copy.deepcopy(net)

    # @classmethod
    # def build_from_cfgs(self, net_cfg, learner_cfg, *args, **kwargs):
    #     net = InfoNet(*args, cfg=net_cfg, **kwargs)
    #     return InfoLearner(net, cfg=learner_cfg)

    def get_coef(self):
        return self.coef * self.info_decay.get()

    def intrinsic_reward(self, traj):
        info_reward = self.net(traj, mode='reward')
        return 'info', info_reward * self.get_coef()
    
    def update(self, rollout, update_with_latent=True):
        if update_with_latent  == self.use_latent and not self.learn_with_dynamics:
            z_detach = rollout['z'].detach()

            mutual_info = self.net(rollout, mode='likelihood').mean()
            #if self.learn_posterior:
            #    posterior = self.net.get_posterior(rollout['state'][1:].detach(), z_detach).mean()
            #else:
            #    posterior = 0.
            posterior = 0.

            self.optimize(- mutual_info - posterior)

            # TODO: estimate the posterior
            logger.logkv_mean('info_ce_loss', float(-mutual_info))
            # logger.logkv_mean('info_posterior_loss', float(-posterior))

            self.info_decay.step()
            logger.logkv_mean('info_decay', self.info_decay.get())


    def update_batch(self, seg):
        if not self.use_latent and not self.learn_with_dynamics:
            self.update(seg2rollout(seg), update_with_latent=False)


    def sample_latent(self, states):
        # raise NotImplementedError
        # assert not self.learn_with_dynamics
        # return self.net(
        #     {
        #         'state': states,
        #         'a': a[None,:],
        #         'z': a[None, :] * 0 # fake z
        #     },
        #     mode='sample'
        # )[0]
        #return self.net({'state': [states, states]})
        # state based mutual information
        feature = self.target_net.compute_feature(states, None)
        return self.net.hidden.sample(feature)[0] # do not need the log prob ..

    # def sample_posterior(self, states):
    #     assert not self.learn_with_dynamics

    #     return self.net.get_posterior(states)[0]

    def ema(self, decay=0.999):
        from tools.utils import ema
        ema(self.net, self.target_net, decay)