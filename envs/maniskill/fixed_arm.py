import numpy as np
from .open_carbinet import OpenCabinetDoorEnv, MobilePandaSingleArm
from mani_skill2.agents.robots.panda import Panda
from typing import Dict, Union


class FixArm(OpenCabinetDoorEnv): 
    # return information about the environment
    EMBED_INPUT_DIM=7

    def __init__(self, *args, obs_dim=8, **kwargs):
        config = {
            'reward_mode': 'dense', 'obs_mode': 'state', 'model_ids': '1018', 'fixed_target_link_idx': 1
        }
        config.update(kwargs)


        self.obs_dim = obs_dim
        if self.obs_dim > 0:
            #from ..maze import get_embedder
            from ..utils import get_embeder_np
            self.embedder, _ = get_embeder_np(self.obs_dim, self.EMBED_INPUT_DIM)

        super().__init__(*args, obs_dim=obs_dim, **config)

        self.joints = self.agent.robot.get_active_joints()
        assert len(self.agent.robot.get_qpos()) == len(self.joints)
        print([j.get_name() for j in self.joints], len(self.joints))

        qpos = self.agent.robot.get_qpos()
        for i in range(4):
            self.joints[i].set_limits([[qpos[i]-1e-5, qpos[i] + 1e-5]])

        self.sample_anchor_points()
            


    def _initialize_robot(self):
        # Base position
        # The forward direction of cabinets is -x.
        center = np.array([1 - 0.5, 0.0])
        #dist = self._episode_rng.uniform(1.6, 1.8)
        #theta = self._episode_rng.uniform(0.9 * np.pi, 1.1 * np.pi)
        dist = 1.6
        # theta = 1.
        theta = np.pi
        direction = np.array([np.cos(theta), np.sin(theta)])
        xy = center + direction * dist

        # Base orientation
        ori = 0.
        h = 1e-4
        arm_qpos = np.array([0, 0, 0, -1.5, 0, 3, 0.78, 0.02, 0.02])

        qpos = np.hstack([xy, ori, h, arm_qpos])
        self.agent.reset(qpos)

    def obs2qpos(self, obs):
        qpos = np.concatenate((obs[..., 20:23], obs[..., :10]), axis=-1)
        return qpos
        
    def wrap_obs(self, obs):
        if self.obs_dim > 0:
            qpos = self.obs2qpos(obs)
            qpos = qpos[4:-2] # remove the first 4 x, y, ori, height, and the last two fingers dim..
            obs = np.concatenate((obs*0.05, self.embedder(qpos)))
        else:
            obs = obs

        return obs

    def decode_obs_to_qpos(self, obs):
        qpos = self.obs2qpos(obs)
        if self.obs_dim > 0:
            qpos = qpos / 0.05
        return qpos

    def sample_anchor_points(self, N=10000):
        state = np.random.get_state()
        np.random.seed(0)
        qlimits = np.array([i.get_limits() for i in self.joints])[:, 0]
        anchors = []
        for i in range(len(qlimits)):
            random_qpos = np.random.uniform(qlimits[i, 0], qlimits[i, 1], (N,))
            anchors.append(random_qpos)
        self.anchors = np.stack(anchors, axis=1) 
        np.random.set_state(state)

    def show_random_anchor(self):
        x = self.anchors[np.random.randint(len(self.anchors))]
        self.agent.robot.set_qpos(x)
        return self.render(mode='rgb_array')

    def _load_agent(self):
        self.agent = MobilePandaSingleArm(
            self._scene, 
            self._control_freq, 
            self._control_mode, 
            fix_root_link=True
        )


    def get_obs_from_traj(self, traj):
        if isinstance(traj, dict):
            obs = traj['next_obs']
            import torch
            if isinstance(obs, torch.Tensor):
                obs = obs.detach().cpu().numpy()
        else:
            obs = traj.get_tensor('next_obs', device='numpy')
        return obs

    def _render_traj_rgb(self, traj, occ_val=False, history=None, verbose=True, **kwargs):
        obs = self.get_obs_from_traj(traj)
        qpos = self.decode_obs_to_qpos(obs)
        qpos = qpos.reshape(-1, qpos.shape[-1])[..., 4:-2]

        anchor_pos = self.anchors[..., 4:-2]

        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(qpos)
        occupancy, _ = nbrs.kneighbors(anchor_pos)

        if occ_val >= 0:
            if history is not None:
                occupancy = np.minimum(history['occ'], occupancy)
                
        else:
            occupancy = None

        if verbose:
            import matplotlib.pyplot as plt
            from tools.utils import plt_save_fig_array
            plt.clf()
            plt.hist(occupancy, bins=100)
            img = {'hist': plt_save_fig_array()}
        else:
            img = {}
            
        output = {
            # 'state': obs,
            'background': {},

            'image': img,
            'history': {
                'occ': occupancy,
            },
            'metric': {'occ': occupancy.mean(),}
        }

        return output

    
    # def step(self, action: Union[None, np.ndarray, Dict]):
    #     #raise NotImplementedError
    #     obs, reward, done, info = super().step(action)
    #     return self.wrap_obs(obs), reward, done, info

    # def reset(self, seed=None, reconfigure=False):
    #     obs = super().reset(seed, reconfigure)
    #     return self.wrap_obs(obs)