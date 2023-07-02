import cv2 
import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from .goal_env_base import GoalEnv

class StateEnv(GoalEnv):
    def __init__(self, cfg=None):
        super().__init__()
        self.state = None
        self.goal = None

    def reset_state_goal(self, states, goals):
        assert len(states.shape) == 2, f"{states.shape}"
        self.state = torch.tensor(states, device='cuda:0', dtype=torch.float32)
        self.goal = goals
        assert self.state.dim() == 2

    def get_obs(self):
        return self.state.clone()

    def get_state(self):
        return self.state.detach().cpu().numpy()

    @property
    def batch_size(self):
        return len(self.state)


class PointEnv(StateEnv):
    # let's consider the penalty based version, to ensure the environment is continuous  

    def __init__(self, cfg=None, low_steps=15, num_stages=1, with_dt=False, penalty=False, boundary=True, clamp_action=True, save_traj=False, action_scale=0.1, action_penalty=1.):
        super().__init__()
        self.action_scale = action_scale
        self.observation_space = gym.spaces.Box(-1, 1, (3,))
        self.action_space = gym.spaces.Box(-1, 1, (2,))
        self.state = torch.zeros(3, dtype=torch.float32, device='cuda:0')
        self.traj_idx = 0

    def sample_state_goal(self, batch_size=1):
        state = np.zeros((batch_size, 3))
        state[:, 1] = 0.5
        return state, np.zeros((batch_size, 3))

    def obstacle(self, target):
        #x, y = target.detach().cpu().numpy()
        x = target[:, 0]
        y = target[:, 1]
        return torch.logical_and(torch.logical_and(x>0.3, y>=0.3), y<=0.7)

    def penalty(self, x):
        # 0 if x < 0 else 
        # if x > 0.1, x will be e ..
        return torch.tanh(torch.relu(x/0.1))

    def get_obstacles(self):
        self.obstacle_penalty = self.penalty( 0.4 - ((self.state[:, :2] - 0.5)**2).sum(axis=1) ** 0.5 ) 

    def obstalce_impulse(self):
        raise NotImplementedError

    def get_reward(self):
        return self.state[:, 0]

    # use action penalty
    def step(self, action):
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(np.array(action), dtype=torch.float32, device='cuda:0')

        if self._cfg.clamp_action:
            action = action.clamp(-1, 1)
        action = torch.cat((
            #action.clamp(-1., 1.),
            action,
            torch.ones_like(action[:, :1]) * self._cfg.with_dt * 0.1), -1)
        self.state = self.state + action * self.action_scale

        #in_obstacle = self.obstacle(target).float()
        #self.state = target.clamp(0., 1.) * (1-in_obstacle) + self.state
        reward = self.get_reward()

        #reward -= self.state - torch.stack()
        info = {}

        # receive very large penalty if |action| > 1
        self.action_penalty = self.penalty((torch.abs(action)-0.9)/2.).sum(axis=-1)  * self._cfg.action_penalty

        info['action_penalty'] = self.action_penalty.mean().item()
        if self._cfg.penalty:
            self.get_obstacles()

            # can not be more than 1, smaller than 0, or in the circle ..
            self.boundary_penalty = self.penalty(-self.state[:, :2]).sum(axis=-1) + self.penalty(self.state[:, :2] - 1).sum(axis=-1) 
            info['obstacle_penalty'] = self.obstacle_penalty.mean().item()
        else:
            self.obstacle_penalty = 0.
            self.boundary_penalty = 0.
            info = {}

            # TODO: Replace to differentiable clamp .. 
            if self._cfg.boundary:
                self.state[:, :2] -= (self.state[:, :2] <= 0).float() * self.state[:, :2]
                self.state[:, :2] -= (self.state[:, :2] - 1 >= 0).float() * (self.state[:, :2] - 1)

            self.obstalce_impulse()

        reward = reward - self.obstacle_penalty * 5 - self.action_penalty * 20 - self.boundary_penalty * 5
        return self.state.clone(), reward, False, info


    def _render_rgb(self, index=0):
        # only render the first ..
        size = 128
        start = size//2
        x, y = self.state[index][:2].detach().cpu().numpy()
        img = np.zeros((size*2, size*2, 3), dtype=np.uint8)
        img = cv2.circle(img, (int(x*size)+start, int(y*size)+start), 3, (255, 0, 0), -1)
        img = cv2.circle(img, (int(0.5*size)+start, int(0.5*size)+start), int(0.4 * size), (0, 255, 0), -1)
        img = cv2.rectangle(img, (start, start), (size+start, size+start), (255, 0, 255), 1, 1)
        return img

        
    def _render_traj_rgb(self, traj, **kwargs):

        states = traj.get_tensor('obs', device='cpu')
        states = states.detach().cpu().numpy() * 128 + 64 
        from tools.utils import plt_save_fig_array
        states = states[:, :, :2]
        return {
            'state': states,
            'background': {
                'image': self._render_rgb()/255.,
                'xlim': [0, 256],
                'ylim': [0, 256],
            },
            'actions': traj.get_tensor('a', device='cpu').detach().cpu().numpy(),
        }


        plt.clf()
        img = self._render_rgb()/255.

        z = traj.get_tensor('z', device='cpu')

        from ..draw_utils import plot_colored_embedding
        plt.clf()

        if z.dtype == torch.int64:
            print(torch.bincount(z.long().flatten()))
        plt.imshow(np.uint8(img[...,::-1]*255))
        plot_colored_embedding(z, states[:, :, :2], s=2)

        plt.xlim([0, 256])
        plt.ylim([0, 256])
        img2 = plt_save_fig_array()[:, :, :3]

        a = traj.get_tensor('a', dtype=None)
        plt.clf()
        plot_colored_embedding(z, a)
        from tools.utils import logger
        logger.savefig('z-a.png')
        return img2

class PointEnv2(PointEnv):
    def __init__(self, cfg=None):
        super().__init__()

    def get_obstacles(self):
        x = self.state[:, :2]

        a = (x[:, 0] - 0.25)**2 + (x[:, 1] - 0.25)**2
        b = (x[:, 0] - 0.25)**2 + (x[:, 1] - 0.75)**2
        c = (x[:, 0] - 0.75)**2 + (x[:, 1] - 0.25)**2
        d = (x[:, 0] - 0.75)**2 + (x[:, 1] - 0.75)**2
        self.obstacle_penalty = self.penalty((0.15 - torch.stack([a, b, c, d]).min(axis=0).values ** 0.5)*4)

    def obstalce_impulse(self):
        for cx, cy in zip([0.25, 0.25, 0.75, 0.75], [0.25, 0.75, 0.25, 0.75]):
            dx = self.state[:, 0] - cx
            dy = self.state[:, 1] - cy
            d = ((dx ** 2) + (dy ** 2)+1e-9)**0.5
            gap = 0.15 - d

            self.state[:, 0] += (gap > 0).float() * dx / d * (0.15 - d)
            self.state[:, 1] += (gap > 0).float() * dy / d * (0.15 - d)

    def sample_state_goal(self, batch_size=1):
        state = np.zeros((batch_size, 3))
        state[:] = 0.01
        return state, np.zeros((batch_size, 3))

    def _render_rgb(self, index=0):
        # only render the first ..
        size = 128
        start = size//2
        x, y = self.state[index][:2].detach().cpu().numpy()
        img = np.zeros((size*2, size*2, 3), dtype=np.uint8)
        img = cv2.circle(img, (int(0.25*size)+start, int(0.25*size)+start), int(0.15 * size), (0, 255, 0), -1)
        img = cv2.circle(img, (int(0.25*size)+start, int(0.75*size)+start), int(0.15 * size), (0, 255, 0), -1)
        img = cv2.circle(img, (int(0.75*size)+start, int(0.75*size)+start), int(0.15 * size), (0, 255, 0), -1)
        img = cv2.circle(img, (int(0.75*size)+start, int(0.25*size)+start), int(0.15 * size), (0, 255, 0), -1)
        img = cv2.rectangle(img, (start, start), (size+start, size+start), (255, 0, 255), 1, 1)
        img = cv2.circle(img, (int(x*size)+start, int(y*size)+start), 3, (255, 0, 0), -1)
        return img

    def get_reward(self):
        return self.state[:, 0] + self.state[:, 1]

        
class PointEnv3(PointEnv2):
    def __init__(self, cfg=None):
        super().__init__()
    def get_obstacles(self):
        #self.obstacle_penalty = self.state[:, 0]*0
        super().get_obstacles()
        self.obstacle_penalty = self.obstacle_penalty * 5