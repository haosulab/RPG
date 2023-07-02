import torch
from .goal_env_base import GoalEnv
from .point_env import StateEnv
from tools import merge_inputs
import matplotlib.pyplot as plt
import numpy as np
import gym
import torch
from tools.utils import plt_save_fig_array

class Bandit(StateEnv):
    param_lists = {
        'typeA': {
            'centers': [
                [-0.2, 0.],
                [-0.15, 0.],
                [0.15, 0.],
                [0.7, 0.],
            ],
            'stds': 0.08,
            'heights': [0.21, 0.21, 0.2, 0.23],
            'merge': 'max'
        },
        'two_mode_1d': {
            'centers': [[-0.6], [0.6]],
            'stds': [1/np.sqrt(2), 1/np.sqrt(6)],
            'heights': [0., 0.],
            'merge': 'max'
        },

        'discontinuous': {
            'centers': [[-0.6], [0.6]],
            'stds': [1/np.sqrt(2), 1/np.sqrt(6)],
            'heights': [-0.3, 0.],
            'merge': 'select'
        },

        'discontinuous2': {
            'centers': [[-0.6], [0.6]],
            'stds': [1/np.sqrt(2), 1/np.sqrt(6)],
            'heights': [0., -0.3],
            'merge': 'select'
        },
    }

    def __init__(
        self,
        cfg=None,
        dist_name='two_mode_1d',
        plot_size=128,
        low_steps=1,
        exp_reward=False,

        centers=None, heights=None, stds=None
    ):
        super().__init__()

        self.get_fn(dist_name)

    def get_fn(self, name):
        #hack .. 
        params = self.param_lists[self._cfg.dist_name]
        centers = params['centers'] or self._cfg.centers
        stds = params['stds'] or self._cfg.stds
        heights = params['heights'] or self._cfg.heights
        method = params['merge']

        self.centers = torch.tensor(centers, device='cuda:0')
        self.heights = 0. if heights is None else torch.tensor(np.array(heights), device='cuda:0')
        self.std = 0.2 if stds is None else torch.tensor(np.array(stds), device='cuda:0')

        def calc_reward(x):
            dist = ((x[..., None, :] - self.centers[None, :, :]) ** 2).sum(dim=-1)
            r = - dist / self.std/self.std/2
            if self._cfg.exp_reward:
                r = torch.exp(r)
            r = r  + self.heights
            if method == 'max':
                r = r.max(dim=-1).values
            else:
                flag = (x[:, 0] < 0.).float()
                r = r[..., 0] * flag + r[..., 1] * (1-flag)
            return r
        self.fn = calc_reward

        dim = self.centers.shape[-1]

        self.dim = dim
        self.observation_space = gym.spaces.Box(-1, 1, (dim,))
        self.action_space = gym.spaces.Box(-1, 1, (dim,))

        
    def sample_state_goal(self, batch_size=1):
        return np.zeros((batch_size, self.dim)) + 0.5, np.zeros((batch_size, self.dim))

    def step(self, action):
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(np.array(action), dtype=torch.float32, device='cuda:0')
        #action = action.clamp(-1., 1.)
        assert action.shape == self.state.shape, f"{action.shape}, {self.state.shape}"
        reward = self.fn(action)

        self.state = action
        self._reward = reward.detach().cpu().numpy()
        return self.state, reward, True, {}

    def prepare_background(self):
        size = self._cfg.plot_size
        if self.dim == 1:
            plt.clf()
            x = np.linspace(0., 1., size)
            x =torch.tensor(x[:, None], device='cuda:0') * 2 - 1
            y = self.fn(x).detach().cpu().numpy()
            plt.plot(x.reshape(-1).detach().cpu().numpy(), y)
            return y
        else:
            x = np.linspace(0., 1., size)
            X,Y = np.meshgrid(x, x) # grid of point
            inp = torch.tensor(np.stack((X, Y), -1), device='cuda:0')
            y = self.fn((inp*2)-1).detach().cpu().numpy() # evaluation of the function on the grid
            y = y.reshape(size, size) 
            y = y - y.min()
            y = y / y.max()
            self.im = np.uint8(np.stack((y*0, y*0.2, y*0.5), axis=-1) * 255)

    def _render_rgb(self, index=0, show_reward=False):
        self.prepare_background()
        if self.dim == 2:
            if isinstance(index, int):
                import cv2 
                x, y = self.state[index]
                import copy
                img = copy.copy(self.im)

                img = cv2.circle(img, (int(x*img.shape[1]), int(y*img.shape[0])), 3, (255, 0, 0), -1)
                if show_reward and hasattr(self, '_reward'):
                    cv2.putText(img, f'r{self._reward[index]:.3f}',
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0., 0, 255.),
                        thickness=2, lineType=cv2.LINE_AA
                    )
        elif self.dim == 1:
            y = self.prepare_background()
            head = y.max()
            low = y.min()
            x = float(self.state[index][0])
            plt.plot([float(x), float(x)], [low, head], color="red", linestyle='dashed')
            img = plt_save_fig_array()

        return img

    def _render_traj_rgb(self, states, traj=None, info=None):
        states = states[1:].reshape(-1, self.dim)
        if self.dim == 1:
            import scipy.stats as ss
            plt.clf()
            self.prepare_background()
            img1 = plt_save_fig_array()

            plt.clf()
            plt.hist([float(i[0]) for i in states], bins=20, density=True)
            plt.xlim([-1.5, 1.5])
            img2 = plt_save_fig_array()
            return np.concatenate((img1, img2), axis=1)
        else:
            raise NotImplementedError


class FourBandit(Bandit):
    def __init__(self, cfg=None, dist_name='typeA'):
        super().__init__()


class TrajBandit(Bandit):
    # penalty to avoid contact .. 
    def __init__(self, cfg=None, low_steps=6):
        super().__init__()

    def step(self, action):
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(np.array(action), dtype=torch.float32, device='cuda:0')
        assert action.shape[-1] == 2
        assert action.shape[0] == self.state.shape[0]
        action = action.clamp(-1., 1.)
        target = (self.state + action * self.action_scale).clamp(0., 1.)
        self.state = target

        score = self.fn(self.state * 2 - 1) * self._cfg.task_reward

        reward = score #- before
        return self.state.clone(), reward, False, {}
