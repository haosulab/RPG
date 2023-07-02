# https://github.com/hzaskywalker/hrl/blob/main/hrl/envs/block/multiblock.py

import numpy as np
import sapien.core as sapien
import gym
from .sapien_sim import SimulatorBase, Pose, load_state_vector, state_vector
from tools.config import Configurable, as_builder
from typing import Optional

import numpy as np
from gym import Env, spaces

from .sapien_sim import SimulatorBase, Pose
from .sapien_utils import add_link, identity, x2y


def sample_blocks(n, block_size, world_size, previous=()):
    objects = list(previous)
    for i in range(n):
        not_found = True
        while not_found:
            not_found = False
            xy = (np.random.random((2,)) * 2 - 1) * world_size
            for j in objects:
                if np.abs(j - xy).min() < block_size:
                    not_found = True
                    break
        objects.append(xy)
    return objects[len(previous):]


COLORS = [
    [0.25, 0.25, 0.75],
    [0.25, 0.75, 0.25],
    [0.25, 0.75, 0.75],
]


def set_actor_xy(actor, xy):
    p = actor.get_pose()
    new_p = p.p
    new_p[:2] = np.array(xy)
    actor.set_pose(Pose(new_p, p.q))


class BlockEnv(gym.Env, SimulatorBase):
    def __init__(
        self, dt=0.01,
        frameskip=8,
        gravity=(0, 0, 0),
        random_blocks=False,
        random_goals=False,
        n_block=3,
        success_reward=0,
        reward_type='sparse',
        obs_dim=5,
    ):

        self.obs_dim = obs_dim
        self.reward_type =reward_type
        assert reward_type == 'sparse'

        
        self.n_block = n_block
        self.blocks = []
        self.goal_vis = []

        self.block_size = 0.2
        self.wall_width = 0.1
        self.world_size = 2 - self.block_size
        self.success_threshold = self.block_size

        self.random_blocks = random_blocks
        self.random_goals = random_goals
        if self.obs_dim > 0:
            from ..utils import get_embedder
            self.emebdder, _ = get_embedder(self.obs_dim)

        super(BlockEnv, self).__init__(dt, frameskip, gravity)


        obs = self.reset()
        self.observation_space = spaces.Box(low=-4, high=4, shape=(len(obs),))
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))
        self.success_reward = success_reward



    def get_agent_pos(self):
        return self.agent.get_qpos()

    def get_agent_vel(self):
        return self.agent.get_qvel()

    def step(self, action: np.ndarray):
        action = np.asarray(action)
        for i in range(self.frameskip):
            self.agent.set_qvel(action.clip(-1, 1) * 4)
            self._scene.step()

        r = self.compute_reward()
        info = {'success': self.success, 'metric': {'true': (self.success == self.n_block)}}
        r = (self.success == self.n_block)
        return self._get_obs(), r, False, info

    def reset(self):
        SimulatorBase.reset(self)

        
        if self.random_blocks:
            xys = sample_blocks(self.n_block, self.block_size, self.world_size, previous=())
            for b, xy in zip(self.blocks, xys):
                b.set_qpos(xy)

        if self.random_goals:
            goals = sample_blocks(self.n_block, self.block_size, self.world_size, previous=())
            for g, xy in zip(self.goal_vis, goals):
                set_actor_xy(g, xy)


        return self._get_obs()

    def render(self, mode='human'):
        return SimulatorBase.render(self, mode)

    

    def get_goals(self):
        goals = []
        for g in self.goal_vis:
            goals.append(g.get_pose().p[:2])
        return np.array(goals)

    def compute_reward(self):
        r = 0
        self.success = 0
        contacts = []

        agent_pos = self.get_agent_pos()
        for b, g in zip(self.blocks, self.goal_vis):
            box_pos = b.get_qpos()
            dist = np.linalg.norm(box_pos - g.get_pose().p[:2])
            self.success += dist < self.success_threshold
            r += dist

            contacts.append( np.linalg.norm(agent_pos - box_pos) )

        contact_reward = np.min(contacts)
        
        return - r  -  contact_reward * 0.3 + (self.success) * self.success_reward  # contact bonus .. 


    def build_scene(self):
        np.random.seed(0)

        wall_color = (0.3, 0.7, 0.3)

        wall_width = self.wall_width
        world_width = self.world_size + self.block_size + self.wall_width
        self.add_box(-world_width, 0, (wall_width, world_width - wall_width, 0.5), wall_color, 'wall1', True, False)
        self.add_box(+world_width, 0, (wall_width, world_width - wall_width, 0.5), wall_color, 'wall2', True, False)
        self.add_box(0, +world_width, (world_width + wall_width, wall_width, 0.5), wall_color, 'wall3', True, False)
        self.add_box(0, -world_width, (world_width + wall_width, wall_width, 0.5), wall_color, 'wall4', True, False)

        material = self._sim.create_physical_material(0, 0, 0)
        self._scene.add_ground(0, material=material)

        world_width = self.world_size
        self.world_size = world_width
        ranges = np.array([[-world_width, world_width], [-world_width, world_width]])
        self.ranges = ranges
        self.material = material

        DEFAULT_GOAL = [
            [-1.4, 1.4],
            [0., 1.4],
            [1.4, 1.4],
        ]

        for i in range(self.n_block):
            self.blocks.append(
                self.add_articulation(
                    (i - 1) * 1.2, 0.0, self.block_size, COLORS[i],
                    ranges, f"block{i}", friction=0,
                    damping=5000, material=material)
            )
            self.goal_vis.append(
                self.add_box(*DEFAULT_GOAL[i],
                             (self.block_size, self.block_size, 0.1),
                             np.array(COLORS[i]) * 1.2, f'goal{i}', True, False))
        self.objects = self.blocks + self.goal_vis

        
        self.agent = self.add_articulation(
            0., -1.4, self.block_size, (0.75, 0.25, 0.25),
            self.ranges, "pusher", friction=0, damping=0, material=self.material)
        self.objects.append(self.agent)


    def count(self, obs):
        import torch
        SIZE = 1.8
        N = 4
        gap = (SIZE * 2 / N) / 2
        # x = torch.linspace(-SIZE, SIZE, N, device=self.device)
        # y = torch.linspace(-SIZE, SIZE, N, device=self.device)

        # x, y = torch.meshgrid(x, y, indexing='ij')
        # anchor =  torch.stack([y, x], dim=-1).cuda()

        def discrete(pos):
            pos = torch.tensor(pos, device='cuda')
            round = ((pos - (-SIZE))/(gap * 2)).long()
            pos = round.clamp(0, N-1)
            return pos[..., 0] * N + pos[..., 1]

        outputs = {}
        index = {k: discrete(v).reshape(-1) for k, v in obs.items()}

        total = 1
        for i in range(self.n_block):
            total = total * (N*N)
        total_ind = 0
        import torch_scatter

        for k, v in index.items():
            assert v.max() < (N*N)
            outputs[k] = torch_scatter.scatter_add(torch.ones_like(v), v, dim=0, dim_size=N*N).view(N, N)
            if k != 'obs':
                total_ind = total_ind * (N*N) + v

        outputs['total'] = torch_scatter.scatter_add(torch.ones_like(total_ind), total_ind, dim=0, dim_size=total)
        return {k: v.detach().cpu().numpy() for k, v in outputs.items()}


    def _render_traj_rgb(self, traj, z=None, occ_val=False, history=None, **kwargs):
        obs = self.get_obs_from_traj(traj)
        if occ_val >= 0:
            occupancy = self.count(obs) 
            if history is not None:
                #occupancy += history['occ']
                for k, v in occupancy.items():
                    occupancy[k] = v + history['occ'][k]
            import copy
            history = {'occ': copy.deepcopy(occupancy)}
            occ_total = occupancy.pop('total')
            occ_total = (occ_total > 0).mean()
        else:
            occupancy = None


        print(occupancy)
        output = {
            'state': obs,
            'background': {
                'image':  None,
                'xlim': [-self.world_size, self.world_size],
                'ylim': [-self.world_size, self.world_size],
                    },
            'image': {k: np.uint8(np.float32(occupancy[k] > 0.) * 255) for k in occupancy},
            'metric': {'occ': occ_total},
            'history': history, 
        }
        return output


    def get_obs_from_traj(self, traj):
        if isinstance(traj, dict):
            obs = traj['next_obs']
        else:
            obs = traj.get_tensor('next_obs')

        if self.obs_dim > 0:
            obs = obs[..., self.original_obs_length:] * 10

        agent_pos = obs[..., -4:-2].reshape(-1, 2)
        start = -4 - self.n_block * 6
        outputs = {}
        outputs['obs'] = agent_pos.detach().cpu().numpy()
        box_pos = obs[..., start:start + self.n_block*2].reshape(agent_pos.shape[0], self.n_block, 2)
        for i in range(self.n_block):
            outputs[f'box{i}'] = box_pos[:, i].detach().cpu().numpy()
        return outputs
        # return {
        #     'obs': ,
        #     'box0': box_pos[:, 0].detach().cpu().numpy(),
        #     'box1': box_pos[:, 1].detach().cpu().numpy(),
        #     'box2': box_pos[:, 2].detach().cpu().numpy(),
        # }

    def _get_obs(self):
        pos, vel, diff = [], [], []
        for b, g in zip(self.blocks, self.goal_vis):
            pos.append(b.get_qpos())
            vel.append(b.get_qvel())
            diff.append(g.get_pose().p[:2] - b.get_qpos())
        assert len(self.get_agent_pos()) == 2
        agent_pos = self.get_agent_pos()
        obs =  np.concatenate(pos + vel + diff + [agent_pos, self.get_agent_vel()])
        self.original_obs_length = len(obs)

        if self.obs_dim > 0:
            obs = obs * 0.1
            pos = pos + [agent_pos]
            out = []
            for i in pos:
                import torch
                out.append(self.emebdder(torch.tensor(i/2)).detach().cpu().numpy())
            out.append(obs)
            obs = np.concatenate(out, axis=-1)

        return obs

