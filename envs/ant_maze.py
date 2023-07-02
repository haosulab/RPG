# ant environment for exploration
import numpy as np
import torch
import gym
from envs.pacman.antman import AntManEnv
#from .maze import get_embedder
from .utils import get_embedder


class AntMaze(gym.Env):
    SIZE = 4
    def __init__(self, obs_dim=8, init_pos=(3, 3), maze_id=0, maze_type=None, lookat=(9, 9, 5), reset_loc=False) -> None:
        super().__init__()
        self.reset_loc = reset_loc
        self.reward = reward
        self.init_pos = init_pos

        self.ant_env = AntManEnv(reset_maze=False, reset_seed=maze_id, maze_type=maze_type, lookat=lookat)

        self.obs_dim = obs_dim
        if self.obs_dim > 0:
            self.embedder, _ = get_embedder(obs_dim)

        self.grid_size = self.ant_env.ant_env.MAZE_SIZE_SCALING
        obs = self.reset()
        self.init_local_obs = self.low_obs.copy()

        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=obs.shape, dtype=np.float32)
        self.action_space = self.ant_env.action_space

        self.action_scale = self.action_space.high[0]
        self.action_space = gym.spaces.Box(-1, 1, shape=self.action_space.shape, dtype=np.float32)

        self.device = 'cuda:0'

        
    def decorate_obs(self, obs):
        if self.obs_dim > 0:
            pos = obs[:2] / self.grid_size / self.SIZE
            obs = np.concatenate(
                [pos/100., obs[2:]/10., self.embedder(obs)])
        return obs

    def get_obs(self):
        return self.decorate_obs(self.low_obs.copy())

    def reset(self):
        if self.reset_loc:
            init_pos = (np.random.randint(4), np.random.randint(4))
        else:
            init_pos = self.init_pos
        self.low_obs = self.ant_env.reset(player=list(init_pos))
        # print(self.reset_loc, init_pos, self.ant_env.loc)
        # print('init pos', init_pos, self.ant_env.loc)
        return self.get_obs()

    def step(self, action):
        # add clip ..
        self.low_obs, reward, _, _ = self.ant_env.step(action.clip(-1., 1.) * self.action_scale)
        return self.get_obs(), reward, False, {}

    def render(self, mode='rgb_array'):
        return self.ant_env.render(mode=mode)

    def get_obs_from_traj(self, traj):
        if isinstance(traj, dict):
            obs = traj['next_obs']
        else:
            obs = traj.get_tensor('next_obs')
        obs = obs[..., :2]
        if self.obs_dim > 0:
            obs = obs * 100 * self.grid_size * self.SIZE
        return obs

    def get_occupancy_image(self, occupancy):
        return occupancy / occupancy.max()

    def get_xlims(self):
        return {
            'xlim': [0, self.grid_size * 4],
            'ylim': [0, self.grid_size * 4],
        }

    def _render_traj_rgb(self, traj, z=None, occ_val=False, verbose=True, history=None, **kwargs):
        obs = self.get_obs_from_traj(traj)

        if occ_val >= 0:
            occupancy = self.counter(obs) 
            if history is not None:
                occupancy += history['occ']
        else:
            occupancy = None

        images = {}
        if verbose:
            images = {'occupancy': self.get_occupancy_image(occupancy)}

        obs = obs.detach().cpu().numpy()
        output = {
            'state': obs,
            'background': {
                'image':  None,
                **self.get_xlims(),
            },
            'image': images,
            'history': {'occ': occupancy},
            'metric': {'occ': (occupancy > occ_val).mean()},
        }

        return output

    def build_anchor(self):
        x = torch.arange(0., 4, device=self.device) + 0.5
        y = torch.arange(0., 4, device=self.device) + 0.5
        x, y = torch.meshgrid(x, y, indexing='ij')
        return torch.stack([y, x], dim=-1).cuda()

    def counter(self, obs):
        anchor = self.build_anchor()
        obs = obs.reshape(-1, obs.shape[-1])/self.grid_size
        #print(obs.shape, anchor.shape)
        reached = torch.abs(obs[None, None, :, :] - anchor[:, :, None, :])
        reached = torch.logical_and(reached[..., 0] < 0.5, reached[..., 1] < 0.5)
        return reached.sum(axis=-1).float().detach().cpu().numpy()[::-1]

    @property
    def anchor_state(self):
        init_obs = self.init_local_obs
        #for x in range(4):
        #    for y in range(4):
        anchor_obs = self.build_anchor().detach().cpu().numpy()
        
        outs = []
        coords = []
        for i in anchor_obs.reshape(-1, 2):
            # print(i * self.grid_size, init_obs[:2])
            coords.append(i * self.grid_size)
            outs.append(
                self.decorate_obs(np.concatenate([i * self.grid_size, init_obs[2:]]))
            )
            # print(outs[-1])
        return {
            'state': np.stack(outs),
            'coord': np.stack(coords),
        }

        
class AntCross(AntMaze):
    # ant maze with four 
    SIZE = 7
    def __init__(self, obs_dim=8, reward=False, init_pos=(0, 0), maze_id=0, maze_type='cross') -> None:
        init_pos = (-0.5, -0.5)
        super().__init__(obs_dim, reward, init_pos, maze_id, maze_type=maze_type, lookat=(0, -5, 10))
        self.anchors = self.ant_env.ant_env.anchors
        self.minL = self.anchors.min()
        self.maxL = self.anchors.max()
        self.range = self.maxL / self.grid_size

    def build_anchor(self):
        return torch.tensor(self.anchors, device=self.device, dtype=torch.float32).cuda()

    def get_xlims(self):
        return {
            'xlim': [self.minL, self.maxL],
            'ylim': [self.minL, self.maxL],
        }

    def get_occupancy_image(self, occupancy):
        #return super().get_occupancy_image(occupancy)
        occupancy = occupancy / occupancy.max()
        # print(occupancy.argmax()
        # print(occupancy.shape, self.anchors.shape)
        # print(occupancy)
        # print(self.anchors[occupancy.argmax()])
        # exit(0)
        anchors = (self.anchors / self.grid_size)
        anchors = anchors.astype(np.int64)
        from solver.draw_utils import plot_grid_point_values
        from tools.utils import plt_save_fig_array
        plot_grid_point_values(anchors, np.float32(occupancy > 0.))
        return plt_save_fig_array()

    def counter(self, obs):
        anchor = self.build_anchor()
        obs = obs.reshape(-1, obs.shape[-1])

        reached = torch.abs(obs[None, :, :] - anchor[:, None, :])
        reached = torch.logical_and(reached[..., 0] < 0.5, reached[..., 1] < 0.5)
        
        return reached.sum(axis=-1).float().detach().cpu().numpy()


class AntFork(AntCross):
    def __init__(self, obs_dim=8, reward=False, maze_type='cross2') -> None:
        super().__init__(maze_type=maze_type)