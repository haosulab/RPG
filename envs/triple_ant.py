# ant 
import gym
from envs.pacman.antman import AntMazeEnv
import numpy as np


class TripleAntEnv(gym.Env):
    def __init__(self, n_goals=3) -> None:
        super().__init__()

        width = 4
        height = 4
        self.n_goals =n_goals
        self.ant_env = AntMazeEnv(4, 4, maze_size_scaling=4.8, wall_size=0.1, lookat=(0, 0, 0))


        self.observation_space = self.ant_env.observation_space
        self.action_space = self.ant_env.action_space


        self._high_background = np.zeros((4, height+1, width+1))
        self.ant_env.set_map(self._high_background)
        self.loc = np.zeros(2)

    def get_obs(self):
        return self.low_obs.copy()

    def reset(self):
        self.ant_env.wrapped_env.init_qpos[:2] = 0. #self.loc * self.ant_env.MAZE_SIZE_SCALING
        self.low_obs = self.ant_env.reset()
        return self.get_obs()

    def step(self, action):
        self.low_obs, _, _, _ = self.ant_env.step(action)

        self.loc = self.low_obs[:2].copy() #/self.ant_env.MAZE_SIZE_SCALING

        goals = np.array(
                [
                    [0                      , 0.8], 
                    [1./2 * 3 ** 0.5 , 1./2],
                    [-1./2 * 3 ** 0.5, 1./2]
                ], 
                # # [1./2 * 3 ** 0.5 , 1./2],
                # # [-1./2 * 3 ** 0.5, 1./2],
                # [0                      , -1.], 
                # [0                      , 1.], 
                # [-1.                      , -0.], 
                # [1.                      , 0.], 
        )[:self.n_goals] * self.ant_env.MAZE_SIZE_SCALING
        dist = np.linalg.norm((self.loc[None, :2] - goals[:, :2]), axis=-1)
        reward = (-dist).max(axis=-1)

        reward += 10 * (dist[1] < 1.)
        return self.get_obs(), reward * 0.2, False, {'success': dist[1] < 1.}

    def render(self, mode='rgb_array'):
        return self.ant_env.render(mode=mode)

        
    def _render_traj_rgb(self, traj, **kwargs):
        import matplotlib.pyplot
        from tools.utils import plt_save_fig_array
        import matplotlib.pyplot as plt
        from solver.draw_utils import plot_colored_embedding
        import torch
        #states = states.detach().cpu().numpy()
        states = traj.get_tensor('obs', device='cpu')
        z = traj.get_tensor('z', device='cpu')

        if z.dtype == torch.float64:
            print(torch.bincount(z.long().flatten()))

        states = states[..., :2]
        plt.clf()
        # plt.imshow(np.uint8(img[...,::-1]*255))
        plot_colored_embedding(z, states[:, :, :2], s=2)

        # plt.xlim([0, 256])
        # plt.ylim([0, 256])
        out = plt_save_fig_array()[:, :, :3]
        return out