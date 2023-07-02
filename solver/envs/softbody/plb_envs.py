import os
import gym
import cv2
import pickle
import tqdm
import torch
import numpy as np
from rl_baseline.envs import BaseEnv, plot_pcd, subsample_pcd

from llm.genetic.loader import load_prog, load_scene
from llm.envs import MultiToolEnv
from tools.utils import totensor


FILEPATH = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(FILEPATH, '.cache')

def sample_points(env: MultiToolEnv, num_points, inds, scale=np.ones(3)):
    sim = env.simulator
    points = np.ones((num_points, 3)) * 5

    remain_cnt = num_points
    pbar = tqdm.tqdm(total=remain_cnt)

    body_pos = sim.states[0].body_pos.download()[inds]
    body_rot = sim.states[0].body_rot.download()[inds]

    while remain_cnt > 0:
        p_samples = (np.random.random((sim.n_particles, 3)) - 0.5)  * scale + body_pos
        sim.states[0].x.upload(p_samples)
        sdf_vals = sim.get_dists(f=0, device="numpy")

        sdf_vals = sdf_vals[:, inds]
        accept_map = sdf_vals <= 0
        accept_cnt = sum(accept_map)

        start = num_points - remain_cnt
        points[start:start + accept_cnt] = p_samples[accept_map][:min(accept_cnt, remain_cnt)]
        remain_cnt -= accept_cnt

        pbar.update(accept_cnt)
        pbar.set_description(f"SDF SAMPLING: {max(0, remain_cnt)}", refresh=True)
    pbar.close()
    assert np.all(points != 5)

    from transforms3d import quaternions
    mat = np.eye(4)
    mat[:3, :3] = quaternions.quat2mat(body_rot)
    mat[:3, 3] = body_pos

    points = np.concatenate([points, np.ones((num_points, 1))], axis=1)
    return (points @ np.linalg.pinv(mat).T)


class PlbEnv(BaseEnv):
    def __init__(self, cfg=None,
        task='box_pick',
        render_mode='plt',
        env_cfg=MultiToolEnv.get_default_config(
            sim_cfg=dict(
                max_steps=100,
                gravity=(0., -0.9, 0.),
                ground_friction=1.,
                n_particles=0
            ),
        ),
        n_softbody=800,
        n_tool=200,
        clear_cache=True,
        low_steps=50,
        dist_weight=1.,
    ):
        super().__init__()
        print(cache_dir)

        os.makedirs(cache_dir, exist_ok=True)
        scene_cache_path = os.path.join(cache_dir, task+'.scene')
        if os.path.exists(scene_cache_path) and not clear_cache:
            self.init_state = torch.load(scene_cache_path)
        else:
            self.init_state = load_scene(os.path.join(FILEPATH, 'configs', task + '.yml'))

        n_particles = len(self.init_state.X) + 10
        n_particles = max(n_particles, env_cfg.sim_cfg.n_particles)
        env_cfg.defrost()
        env_cfg.sim_cfg.n_particles = n_particles


        self.env = MultiToolEnv(cfg=env_cfg)
        self.env.extend(low_steps * 20 + 10)
        self.device = self.env.device

        self.env.set_state(self.init_state)
        self.load_tool_pcd(task)

        self.action_space = self.env.action_space

        obs = self.reset()[0]
        self.observation_space = dict(
            xyz=gym.spaces.Box(-np.inf, np.inf, shape=obs['xyz'].shape),
            rgb=gym.spaces.Box(-np.inf, np.inf, shape=obs['rgb'].shape),
            agent=gym.spaces.Box(-np.inf, np.inf, shape=obs['agent'].shape),
        )

    def load_tool_pcd(self, task):
        self.env.set_state(self.init_state)
        cache_path = os.path.join(cache_dir, task + '_tool.pcd')
        if os.path.exists(cache_path) and not self._cfg.clear_cache:
            tool_pcd = torch.load(cache_path)
        else:
            tool_pcd = []
            for i in range(self.env.tool_cur.n_bodies):
                pcd = sample_points(self.env, 5000, i, scale=np.array([0.3, 0.3, 0.3]))
                tool_pcd.append(totensor(pcd, 'cpu'))
            torch.save(tool_pcd, cache_path)
        self.tool_pcd = [totensor(pcd, device=self.device) for pcd in tool_pcd]

    def sample_state_goal(self, batch_size=1):
        return self.init_state, None

    def reset_state_goal(self, states, goals):
        self.env.set_state(states, requires_grad=True)

    def wrap_action(self, action):
        tool = action.reshape(self.env.action_space.shape) # * self.actor_scale
        return totensor(tool, device=self.device), None

    def get_reward(self, obs):
        return obs[0]['dist'].min(axis=0)[0].sum(axis=-1, keepdim=True) - obs[0]['xyz'][:, 2].mean()

    def step(self, action):
        assert action[0].shape == self.action_space.shape
        from tools.utils import clamp
        self.env.step(clamp(action[0], -2., 2.)) # clamp the actions.
        obs = self.get_obs()

        # should be minimized
        #dist = obs[0]['dist'].min(axis=0)[0].sum(axis=-1, keepdim=True) - obs[0]['xyz'][:, 2].mean()
        dist = self.get_reward(obs)

        self._nsteps += 1
        action_penalty = -(torch.relu(torch.abs(action)-0.9)).sum(axis=-1) * 10.
        if self._nsteps != self._cfg.low_steps:
            dist = 0. # only compute the loss at the last timestep.

        return obs, -dist * self._cfg.dist_weight + action_penalty, False, {}

    def _render_rgb(self, index=0, render_plt=True, ray_tracing=True, **kwargs):
        assert index == 0
        if render_plt or ray_tracing:
            img1 = self.env.render(mode='rgb_array', **kwargs)
        if self._cfg.render_mode == 'plt':
            tmp = self.get_obs()[index]
            pcd = tmp['xyz']
            rgb = tmp['rgb']
            hand = pcd[rgb[:, -3] > 0.5][:, [0, 2, 1]]
            shape = pcd[rgb[:, -2] > 0.5][:, [0, 2, 1]]
            img2 = plot_pcd(hand, shape)
            if render_plt:
                img1 = cv2.resize(img1[..., :3], (256, 256))
                img2 = cv2.resize(img2[..., :3], (256, 256))
                img1 = np.concatenate((img1, img2), axis=1)
            else:
                img1 = img2
        return img1

    def get_obs(self):
        # TODO: needs to be differentiable ..
        # out = self.state_to_scene_particles(self.frame_transforms)
        obs = self.env.get_obs()

        tool_pcd = []
        for state, pcd in zip(obs['tool'], self.tool_pcd):
            pos = state[:3]
            rot = state[3:]

            from pytorch3d.transforms.rotation_conversions import quaternion_to_matrix
            pose = torch.eye(4, device=self.device)
            pose[:3, :3] = quaternion_to_matrix(rot)
            pose[:3, 3] = pos
            tool_pcd.append((pcd @ pose.T)[:,:3])

        tool_pcd = torch.cat(tool_pcd, dim=0)

        out = {
            'shape':  subsample_pcd(self._cfg.n_softbody, obs['pos']),
            'tool': subsample_pcd(self._cfg.n_tool, tool_pcd),
        }

        points = []
        rgbs = []

        for idx, key in enumerate(['tool', 'shape']): # note that shape should have an idx -1
            xyz = out[key]
            rgb = torch.zeros_like(xyz)
            rgb[:, idx] = 1.
            points.append(xyz)
            rgbs.append(rgb)

        out = {
            'rgb': torch.cat(rgbs, 0),
            'xyz': torch.cat(points, 0),
            'dist': obs['dist']
        }
        out['agent'] = obs['qpos'].reshape(-1)
        return [out] # remember that this is for a batch ..


class CutEnv(PlbEnv):
    def __init__(self, cfg=None, task='cut', max_steps=10, low_steps=30, dir=0):
        super().__init__()

    def get_reward(self, obs):

        obs = self.env.get_obs()


        xyz = obs['pos']
        return obs['dist'].min(axis=0)[0].sum(axis=-1, keepdim=True) * 10 + torch.linalg.norm(xyz-self.target, dim=-1, keepdim=True).mean() * 500
        #raise NotImplementedError("the observation is not correct")

    def step(self, action):
        if self._nsteps == 0:
            with torch.no_grad():
                pos = self.init_state.X
                if self._cfg.dir == 0:
                    self.mask = totensor(pos[:, 0] > 0.7, device='cuda:0', dtype=torch.bool)
                    self.target = totensor(pos, device='cuda:0')
                    #self.target[:, 0] -= 0.4
                    self.target[self.mask, 0] += 0.2
                else:
                    self.mask = totensor(pos[:, 0] < 0.3, device='cuda:0', dtype=torch.bool)
                    self.target = totensor(pos, device='cuda:0')
                    #self.target[:, 0] += 0.4
                    self.target[self.mask, 0] -= 0.2
                # import cv2
                # img = plot_pcd(self.target)
                # cv2.imwrite('goal.png', img[..., ::-1])
                # exit(0)


        assert action[0].shape == self.action_space.shape
        from tools.utils import clamp
        self.env.step(clamp(action[0], -2., 2.)) # clamp the actions.
        obs = self.get_obs()

        # should be minimized
        #dist = obs[0]['dist'].min(axis=0)[0].sum(axis=-1, keepdim=True) - obs[0]['xyz'][:, 2].mean()
        dist = self.get_reward(obs)

        self._nsteps += 1
        action_penalty = -(torch.relu(torch.abs(action)-0.9)).sum(axis=-1) * 10.
        if self._nsteps != self._cfg.low_steps:
            dist = 0. # only compute the loss at the last timestep.

        return obs, -dist * self._cfg.dist_weight + action_penalty, False, {}



class RopeEnv(PlbEnv):
    def __init__(self, cfg=None, task='rope', max_steps=10, close_to_right=False, low_steps=50,
            sim_cfg=dict(
                ground_friction=20.,
                 )):
        super().__init__()

    def get_reward(self, obs):
        obs = self.env.get_obs()

        if not self._cfg.close_to_right:
            dist = obs['dist'].min(axis=0)[0].sum(axis=-1, keepdim=True)
        else:
            dist = obs['dist'][self.mask].min(axis=0)[0].sum(axis=-1, keepdim=True)
        xyz = obs['pos']

        left = xyz[self.left]
        right = xyz[self.mask]

        #return  dist - right[:, 2].mean()
        # - torch.linalg.norm(left - self.target_left, dim=-1).mean()
        return dist * 20  + torch.linalg.norm(right - self.target_right, dim=-1).mean()

    def step(self, action):
        if self._nsteps == 0:
            with torch.no_grad():
                pos = self.init_state.X
                self.left = totensor(pos[:, 0] < 0.5, device='cuda:0', dtype=torch.bool)
                self.mask = totensor(pos[:, 0] > 0.65, device='cuda:0', dtype=torch.bool)
                self.target_left = totensor(pos, device='cuda:0')[self.left]
                self.target_right = totensor(pos, device='cuda:0')[self.mask]
                self.target_right[:, 2] += 0.4
                #print(self.mask.sum(), len(pos))
                #exit(0)
                # self.target[torch.logical_not(self.mask), 0] += 0.
                # self.target[self.mask, 0] += 0.2
                # import cv2
                # img = plot_pcd(self.target)
                # cv2.imwrite('goal.png', img[..., ::-1])


        assert action[0].shape == self.action_space.shape
        from tools.utils import clamp
        self.env.step(clamp(action[0], -2., 2.)) # clamp the actions.
        obs = self.get_obs()

        # should be minimized
        #dist = obs[0]['dist'].min(axis=0)[0].sum(axis=-1, keepdim=True) - obs[0]['xyz'][:, 2].mean()
        dist = self.get_reward(obs)

        self._nsteps += 1
        action_penalty = -(torch.relu(torch.abs(action)-0.9)).sum(axis=-1) * 10.
        if self._nsteps != self._cfg.low_steps:
            dist = 0. # only compute the loss at the last timestep.

        return obs, -dist * self._cfg.dist_weight + action_penalty, False, {}