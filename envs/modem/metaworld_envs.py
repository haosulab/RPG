# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections import deque
import numpy as np
import gym
from gym.wrappers import TimeLimit
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE


class MetaWorldWrapper(gym.Env):
    def __init__(self, env, cfg, env_id, obs_dim):
        super().__init__()
        self.env = env
        self.cfg = cfg
        self._num_frames = 1 #cfg.get("frame_stack", 1)
        self.img_size = 224
        self.action_repeat = 2

        self.env_id = env_id
        #assert self.env_id == 'stick-pull'
        

        from envs.utils import get_embeder_np
        self.embedder, d = get_embeder_np(obs_dim, 7) # only for the differences between objects ..

        self._frames = deque([], maxlen=self._num_frames)
        # self.observation_space = gym.spaces.Box(
        #     low=0,
        #     high=255,
        #     shape=(self._num_frames * 3, self.img_size, self.img_size),
        #     dtype=np.uint8,
        # )
        self.action_space = self.env.action_space
        self.camera_name = "corner2"
        self.env.model.cam_pos[2] = [0.75, 0.075, 0.7]
        self.observation_space = gym.spaces.Box(-1, 1, shape=self.reset().shape)

    @property
    def state(self):
        state = self._state_obs.astype(np.float32)
        return np.concatenate((state[:4], state[18 : 18 + 4]))

    def _get_pixel_obs(self):
        return self.render(width=self.img_size, height=self.img_size).transpose(
            2, 0, 1
        )

    def _stacked_obs(self):
        assert len(self._frames) == self._num_frames
        return np.concatenate(list(self._frames), axis=0)

    def _get_obs(self):
        #raise NotImplementedError
        obs =  self._state_obs.copy()
        from ..utils import symlog

        tcp = self.env.tcp_center
        if self.env_id == 'stick-pull':
            stick = obs[4:7]
            handle = obs[11:14]

            inp = np.concatenate((symlog((stick - handle)/0.4), symlog((stick - tcp)/0.4), [obs[3]])) # tcp opened ..
            inp = self.embedder(inp)
        elif self.env_id == 'basketball':
            obj = obs[4:7]
            inp = np.concatenate((symlog((tcp - obj)/0.4), symlog(obj/0.4), [obs[3]])) # tcp opened ..
            inp = self.embedder(inp)
        else:
            raise NotImplementedError

        return np.concatenate((obs * 0.05, inp))

    def reset(self):
        self.env.reset()
        obs = self.env.step(np.zeros_like(self.env.action_space.sample()))[0].astype(
            np.float32
        )
        self._state_obs = obs
        return self._get_obs()

    def step(self, action):
        reward = 0
        for _ in range(self.action_repeat):
            obs, r, _, info = self.env.step(action)
            reward += r
        obs = obs.astype(np.float32)
        self._state_obs = obs
        reward = float(info["success"])
        return self._get_obs(), reward, False, info

    def render(self, mode="rgb_array", width=None, height=None, camera_id=None):
        width = width or self.img_size
        height = height or self.img_size
        return self.env.render(
            offscreen=True, resolution=(width, height), camera_name=self.camera_name
        ).copy()

    def observation_spec(self):
        return self.observation_space

    def action_spec(self):
        return self.action_space

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _render_traj_rgb(self, traj, occ_val=False, history=None, verbose=True, **kwargs):
        # don't count occupancy now ..
        from .. import utils
        high = 0.4
        obs = utils.extract_obs_from_tarj(traj) / 0.05
        stick = obs[..., 4:7]
        handle = obs[..., 11:14]
        outs = dict(occ=utils.count_occupancy(handle - stick, -high, high, 0.02))
        history = utils.update_occupancy_with_history(outs, history)
        output = {
            'background': {},
            'history': history,
            'image': {},
            'metric': {k: (v > 0.).mean() for k, v in history.items()},
        }
        return output


def make_metaworld_env(env_id, obs_dim=8, reward_type='sparse'):
    env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_id + "-v2-goal-observable"]()
    env._freeze_rand_vec = False
    assert reward_type == 'sparse'
    env = MetaWorldWrapper(env, None, env_id, obs_dim)
    return env


if __name__ == '__main__':
    from tools.utils import animate
    #env = make_metaworld_env('stick-pull')
    env = make_metaworld_env('basketball')
    env.reset()

    images = []
    for i in range(1000):
        action = env.action_space.sample()
        done = env.step([1,1,0, 0])[2]
        images.append(env.render('rgb_array'))
        if i % 100 == 0:
            print('123')
            env.reset()
            
    
    animate(images)