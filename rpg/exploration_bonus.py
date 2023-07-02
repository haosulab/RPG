import torch
from tools import Configurable
from tools.utils import totensor, tonumpy
from .buffer import ReplayBuffer, BufferItem
from collections import deque
import numpy as np
from tools.utils import RunningMeanStd
from tools.optim import OptimModule
from .density import DensityEstimator



class ExplorationBonus(Configurable):
    name = 'bonus'
    def __init__(
        self,

        obs_space,
        state_dim, 
        z_space,

        enc_s,
        buffer: ReplayBuffer,

        cfg=None,

        density=DensityEstimator.to_build(TYPE='RND'),

        buffer_size=None,
        update_step=1, update_freq=1, batch_size=512,
        obs_mode='obs',

        scale=0.,

        as_reward=True,
        training_on_rollout=False,
        include_latent=False,
    ) -> None:

        super().__init__()
        if buffer_size is None:
            buffer_size = buffer.capacity

        self.z_space = z_space
        self.density_space = self.make_inp_space(obs_space, state_dim, z_space)
        self.estimator: DensityEstimator = DensityEstimator.build(self.density_space, cfg=density).cuda()


        self.step = 0
        # self.buffer = deque(maxlen=buffer_size)
        # self.bufferz = deque(maxlen=buffer_size)
        self.buffer = BufferItem(buffer_size, {'_shape': obs_space.shape}) #deque(maxlen=buffer_size)
        #self.bufferz = BufferItem(buffer_size, {'_shape': z_space})
        self.bufferz = deque(maxlen=buffer_size)

        self.batch_size = batch_size
        self.enc_s = enc_s
            
        self.as_reward = as_reward
        self.training_on_rollout = training_on_rollout
        self.obs_mode = obs_mode
        self.update_step = update_step
        self.update_freq = update_freq
        
        self.scale = scale

        self.include_latent = include_latent
        if include_latent:
            raise NotImplementedError
            assert self.obs_mode != 'state'
            assert not self.training_on_rollout

    def make_inp_space(self, obs_space, state_dim, hidden_space):
        cfg = self._cfg
        if cfg.obs_mode == 'state':
            inp_dim = state_dim
        elif cfg.obs_mode == 'obs':
            inp_dim = obs_space.shape[0]
        else:
            raise NotImplementedError
        if cfg.include_latent:
            inp_dim += hidden_space.dim
        from gym.spaces import Box
        return Box(-np.inf, np.inf, shape=(inp_dim,))

    def make_inp(self, obs, latent):
        inps = obs
        from tools.utils import totensor
        inps = totensor(inps, device='cuda:0')
        if self.include_latent:
            latent = self.z_space.tokenize(totensor(latent, device='cuda:0'))
            inps = torch.cat([inps, latent], dim=-1)
        return inps


    def add_data(self, data, prevz):
        if not self.training_on_rollout:
            self.buffer.append(totensor(data, device='cuda:0'))
            with torch.no_grad():
                for z in prevz:
                    #self.buffer.append(totensor(i, device='cuda:0'))
                    self.bufferz.append(z)

            self.step += 1
            if self.step % self.update_freq == 0 and len(self.buffer) > self.batch_size:
                for _ in range(self.update_step):
                    data = self.sample_data()
                    self.update(*self.sample_data())

    def process_obs(self, obs):
        if self.obs_mode == 'state':
            return self.enc_s(obs)
        else:
            return obs

    def sample_data(self):
        #pass
        assert not self.training_on_rollout
        index = np.random.choice(len(self.buffer), self.batch_size)
        #obs = [self.buffer[i] for i in index]
        #obs = totensor(obs, device='cuda:0')
        obs = self.buffer[totensor(index, device='cuda:0', dtype=None)]
        if self.include_latent:
            latent = [self.bufferz[i] for i in index]
            latent = totensor(latent, device='cuda:0', dtype=None)
        else:
            latent = None
        obs = self.process_obs(obs)
        return obs, latent

    def visualize_transition(self, transition):
        attrs = transition.get('_attrs', {})
        if 'r' not in attrs:
            from tools.utils import tonumpy
            attrs['r'] = tonumpy(transition['r'])[..., 0] # just record one value ..

        if self.obs_mode == 'state':
            attrs['bonus'] = tonumpy(self.compute_bonus(transition['next_state'], transition['z']))
        else:
            attrs['bonus'] = tonumpy(self.compute_bonus(transition['next_obs'], transition['z']))
        transition['_attrs'] = attrs


    # for training on state
    def update_by_rollout(self, rollout):
        #raise NotImplementedError
        if self.training_on_rollout:
            assert self.obs_mode == 'state'
            self.update(rollout['state'][1:].detach())

    def intrinsic_reward(self, rollout, latent):
        if not self.as_reward:
            assert self.obs_mode == 'state'
            bonus = self.compute_bonus(rollout['state'][1:])
            return self.name, bonus * self.scale
        else:
            # rollout is the obs
            bonus = self.compute_bonus_by_obs(rollout, latent)
            return bonus * self.scale
            

    def update(self, inp, latent) -> torch.Tensor: 
        inp = self.make_inp(inp, latent)
        return self.estimator.update(inp)

    def compute_bonus(self, inp, latent) -> torch.Tensor:
        inp = self.make_inp(inp, latent)
        return -self.estimator.log_prob(inp)

    def compute_bonus_by_obs(self, obs, latent):
        obs = self.process_obs(obs)
        return self.compute_bonus(obs, latent)
        