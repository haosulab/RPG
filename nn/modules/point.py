import torch
import gym
from tools.utils import dshape
from tools.nn_base import Network
from .mlp import LinearMLP
from torch import nn
import numpy as np
from .backbone import Backbone
from gym.spaces import Discrete, Box


class PointNetBase(Network):
    def __init__(
        self, inp_dim, oup_dim,
        cfg=None,
        point_mlp=LinearMLP.get_default_config(mlp_spec=[128, 256]),
        global_mlp=LinearMLP.get_default_config(mlp_spec=[]),
        feature_dim=512,
        max_mean_mix_aggregation=True,
        norm_cfg=None,
    ):
        super().__init__(cfg)
        self.point_mlp = LinearMLP(inp_dim, feature_dim, cfg=point_mlp, norm_cfg=norm_cfg)
        self.global_mlp = LinearMLP(feature_dim, oup_dim, global_mlp)
        self._output_shape = (oup_dim,)

    def forward(self, pcd: torch.Tensor, return_pn_fea=False):
        point_feature = self.point_mlp(pcd)
        if self._cfg.max_mean_mix_aggregation:
            max_part, avg_part = torch.chunk(point_feature, 2, dim=-1)
            feature = torch.cat([max_part.mean(axis=-2),  avg_part.mean(axis=-2)], -1)
        else:
            feature = point_feature.max(axis=-2)
        global_fea = self.global_mlp(feature)
        if not return_pn_fea:
            return global_fea
        else:
            return global_fea, point_feature


class PointNet(Backbone):
    def __init__(
        self,
        space_o,
        cfg=None,
        pn_cfg=PointNetBase.get_default_config(),

        agent_mlp=LinearMLP.get_default_config(mlp_spec=[]),
        fuse_mlp=LinearMLP.get_default_config(mlp_spec=[512]),
        hidden_dim=None,
        output_dim=None,
    ):
        super().__init__(space_o)
        inp_dim = self.obs_space['xyz'].shape[-1] + self.obs_space['rgb'].shape[-1]

        hidden_dim = hidden_dim or pn_cfg.point_mlp.mlp_spec[-1]
        self.pn = PointNetBase(inp_dim, hidden_dim, cfg=pn_cfg)

        inp_action_dim = self.action_space.shape[-1] if self.action_space is not None else 0

        self.agent_mlp = LinearMLP(
            inp_action_dim + self.obs_space['agent'].shape[-1], hidden_dim, cfg=agent_mlp)

        output_dim = output_dim or hidden_dim
        self.fuse_mlp = LinearMLP(hidden_dim * 2, output_dim, cfg=fuse_mlp)
        self._output_shape = (output_dim,)

    def forward(self, x, *, timestep=None):
        x, a = self.preprocess(x, timestep=timestep)

        x = self.batch_input(x)
        pcd = torch.cat((x['xyz'], x['rgb']), dim=-1)
        agent = x['agent']

        pcd_feature = self.pn(pcd)  # ponit net

        agent = torch.cat((agent, a), dim=-1) if a is not None else agent
        a_feature = self.agent_mlp(agent)  # agent net
        feature = torch.cat([pcd_feature, a_feature], dim=-1)
        return self.fuse_mlp(feature)


if __name__ == '__main__':
    from tools.utils import animate
    from solver.envs.softbody import PlbEnv

    env = PlbEnv(task='box_pick')

    pn = PointNet(env.observation_space, None).to('cuda:0')

    images =[]
    obs = env.reset()
    for i in range(4):
        action = env.action_space.sample()
        action[1] = -1.
        obs += env.step([action])[0]

    pn = PointNet((env.observation_space, Discrete(4)), None).to('cuda:0')
    z = torch.randint(0, 4, (len(obs),))
    print(pn((obs, z)).shape)