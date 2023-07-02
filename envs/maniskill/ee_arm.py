# robot arm that focuses on the end effector position + orientation if necessary
# count occupancy separately ..
import numpy as np
from .fixed_arm import FixArm
import sapien.core as sapien

class EEArm(FixArm):
    EMBED_INPUT_DIM=3 + 2 # ee xyz and cabinet position ..

    def __init__(self, *args, obs_dim=8, **kwargs):
        super().__init__(*args, obs_dim=obs_dim, **kwargs)

    def wrap_obs(self, obs):
        #return super().wrap_obs(obs)
        ee_xyz = self.agent.get_ee_coords().mean(axis=0)
        cabinet_xy = self.cabinet.get_qpos()

        state = np.concatenate((ee_xyz, cabinet_xy))
        if self.obs_dim > 0:
            obs = np.concatenate((state * 0.05, obs*0.05, self.embedder(state)))
        else:
            pass
        return obs

    def decode_obs(self, obs):
        if self.obs_dim > 0:
            obs = obs / 0.05
        return obs[..., :5]


    def count_occupancy_xyz(self, state=None):
        # only consider the range before the cabinet
        # x: -1. 0.
        # y: -1. -1.
        # z: 0. 0.6
        gap = 0.05
        low = np.array([-1., -1., 0.])
        high = np.array([0., 1., 0.6])


        low_q = 0.
        high_q = 1.6713272

        outs = {}
        if state is not None:
            from ..utils import count_occupancy
            xyz = state[..., :3]
            outs['xyz'] = count_occupancy(xyz, low, high, gap)
            outs['occ'] = count_occupancy(state[..., 3:5], low_q, high_q, 0.1) # cabinet is the most important
        return outs
        

    def _render_traj_rgb(self, traj, occ_val=False, history=None, verbose=True, **kwargs):
        import torch
        from ..utils import update_occupancy_with_history

        obs = self.decode_obs(self.get_obs_from_traj(traj))
        occupancy = self.count_occupancy_xyz(obs)
        if occ_val >= 0:
            history = update_occupancy_with_history(occupancy, history)

        images = {}
        if verbose:
            from tools.utils import plt_save_fig_array
            import matplotlib.pyplot as plt

            xyz = obs[..., :3]

            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(xyz[..., 0], xyz[..., 1], xyz[..., 2], cmap='jet')

            ax.set_xlim(-1.5, 1.)
            ax.set_ylim(-2., 2)
            ax.set_zlim(0, 2)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            images['traj'] = plt_save_fig_array(fig)


        output = {
            'background': {},
            'history': history,
            'image': images,
            'metric': {k: (v > 0.).mean() for k, v in history.items()}
        }

        return output
