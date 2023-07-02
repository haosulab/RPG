# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections import deque
import torch
import numpy as np
import gym
from gym.wrappers import TimeLimit
import envs.modem.mj_envs.envs.hand_manipulation_suite


class AdroitWrapper(gym.Env):
    def __init__(self, env, cfg, obs_dim):
        super().__init__()
        self.env = env
        self.obs_dim = obs_dim
        self.embedder = None
        
        self.env = env
        self.cfg = cfg
        self._num_frames = cfg.get("frame_stack", 1)
        self._frames = deque([], maxlen=self._num_frames)
        self.img_size = cfg.get('img_size', 84)
        # self.observation_space = gym.spaces.Box(
        #     low=0,
        #     high=255,
        #     shape=(self._num_frames * 3, img_size, img_size),
        #     dtype=np.uint8,
        # )
        self.observation_space = gym.spaces.Box(-1, 1, self.reset().shape)
        self.action_space = self.env.action_space
        self.camera_name = cfg.get("camera_view", "view_1")

    @property
    def state(self):
        return self._state_obs.astype(np.float32)

    def _get_state_obs(self, obs):
        task = self.cfg.get('task')
        from envs.utils import get_embeder_np

        if task == "adroit-door":
            if self.embedder is None:
                self.embedder, d = get_embeder_np(self.obs_dim, 9)

            qp = self.env.data.qpos.ravel()
            palm_pos = self.env.data.site_xpos[self.env.grasp_sid].ravel()
            

            inp = np.concatenate([[obs[-11]], obs[-7:-4], obs[-4:-1]]) # door pos, handle pos, relative pos

            manual = np.concatenate([qp[1:-2], palm_pos])
            obs = np.concatenate([obs[:27], obs[29:32]])
            assert np.isclose(obs, manual).all()
            obs = np.concatenate((obs * 0.05, self.embedder(inp)))
            return obs
        elif task == "adroit-hammer":
            if self.embedder is None:
                self.embedder, d = get_embeder_np(self.obs_dim, 9)

            qp = self.env.data.qpos.ravel()
            qv = np.clip(self.env.data.qvel.ravel(), -1.0, 1.0)
            palm_pos = self.env.data.site_xpos[self.env.S_grasp_sid].ravel()
            manual = np.concatenate([qp[:-6], qv[-6:], palm_pos])
            #obs = obs[:36]
            assert np.isclose(obs[:36], manual).all()

            poses = self.env.get_object_poses()
            inp = np.concatenate([poses[i] for i in ['palm_pos', 'tool_pos', 'target_pos']])
            obs = np.concatenate((inp * 0.05, obs * 0.05, self.embedder(inp)))
            return obs
        elif task == "adroit-pen":
            qp = self.env.data.qpos.ravel()
            desired_orien = (
                self.env.data.site_xpos[self.env.tar_t_sid]
                - self.env.data.site_xpos[self.env.tar_b_sid]
            ) / self.env.tar_length
            manual = np.concatenate([qp[:-6], desired_orien])
            obs = np.concatenate([obs[:24], obs[-9:-6]])
            assert np.isclose(obs, manual).all()
            return obs
        elif task == "adroit-relocate":
            qp = self.env.data.qpos.ravel()
            palm_pos = self.env.data.site_xpos[self.env.S_grasp_sid].ravel()
            target_pos = self.env.data.site_xpos[self.env.target_obj_sid].ravel()
            manual = np.concatenate([qp[:-6], palm_pos - target_pos])
            obs = np.concatenate([obs[:30], obs[-6:-3]])
            assert np.isclose(obs, manual).all()
            return obs
        raise NotImplementedError()

    def _get_pixel_obs(self):
        img_size = int(self.img_size)
        return self.render(width=img_size, height=img_size).transpose(
            2, 0, 1
        )

    def _stacked_obs(self):
        assert len(self._frames) == self._num_frames
        return np.concatenate(list(self._frames), axis=0)

    def reset(self):
        obs = self.env.reset()
        self._state_obs = self._get_state_obs(obs)
        #obs = self._get_pixel_obs()
        #for _ in range(self._num_frames):
        #    self._frames.append(obs)
        #return self._stacked_obs()
        return self._state_obs.copy()

    def step(self, action):
        reward = 0
        for _ in range(self.cfg.get('action_repeat', 2)):
            obs, r, _, info = self.env.step(action)
            reward += r
        self._state_obs = self._get_state_obs(obs)
        #obs = self._get_pixel_obs()
        #self._frames.append(obs)
        info["success"] = info["goal_achieved"]
        # reward = float(info["success"]) - 1.0
        # return self._stacked_obs(), reward, False, info
        # 1 for sparse reward ..
        return self._state_obs.copy(), reward/2 + 1, False, info

    def render(self, mode="rgb_array", width=None, height=None, camera_id=None):
        img_size = self.img_size #self.cfg.get('img_size', 224)
        width = width or img_size
        height = height or img_size
        return np.flip(
            self.env.env.sim.render(
                mode="offscreen",
                width=width,
                height=height,
                camera_name=self.camera_name,
            ),
            axis=0,
        )

    def observation_spec(self):
        return self.observation_space

    def action_spec(self):
        return self.action_space

    def __getattr__(self, name):
        return getattr(self._env, name)


    def _render_traj_rgb(self, traj, occ_val=False, history=None, verbose=True, **kwargs):
        # don't count occupancy now ..
        output = {
            'background': {},
            'history': None,
            'image': {},
            'metric': {}
        }
        return output



# def make_adroit_env(cfg):
#     env_id = cfg.task.split("-", 1)[-1] + "-v0"
#     env = gym.make(env_id)
#     env = AdroitWrapper(env, cfg)
#     env = TimeLimit(env, max_episode_steps=cfg.episode_length)
#     env.reset()
#     cfg.state_dim = env.state.shape[0]
#     return env

def make_adroit_env(env_name, reward_type='sparse', obs_dim=6, img_size=None):
    env = gym.make(env_name, reward_type=reward_type)

    env_type = env_name.split('-')[0]
    env = AdroitWrapper(env, {'frame_stack': 1, 'camera_view': 'view_1', 'img_size': img_size or 84, 'task': f'adroit-{env_type}', 'action_repeat': 2}, obs_dim=obs_dim)
    return env

if __name__ == '__main__':
    # episode length 120
    from tools.utils import animate

    env = make_adroit_env('door-v0')
    env.reset()

    images = []
    for i in range(100):
        r = env.step(env.action_space.sample())[1]
        img = env.render('rgb_array')
        images.append(img)

    animate(images, 'output.mp4')