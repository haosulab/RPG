# critic network
# https://www.notion.so/Model-based-RPG-Q-version-3c97a98eea3445968ef634f684f2d152

# from simplify the code, we only compute $Q$ but not separte the value output..
import torch
from torch import nn
from tools.utils import Seq, mlp, logger
from nn.distributions import CategoricalAction, Normal
from tools.nn_base import Network
from gym import spaces


class BackboneBase(Network):
    def __init__(self, cfg=None, observe_alpha=False, time_embedding=0, backbone_type='mlp') -> None:
        super().__init__()
        assert time_embedding == 0, "time embedding is not supported yet"
        self.observe_alpha = observe_alpha
        self.alpha_dim = 2 if observe_alpha else 0
        self.alpha = None

        self.time_embedding = time_embedding
        self.backbone_type = backbone_type

    def add_alpha(self, *args, timestep=None):
        raise NotImplementedError()
        if self.time_embedding > 0:
            assert timestep is not None
            from .utils import positional_encoding
            args = list(args) + [positional_encoding(self.time_embedding, timestep)]

        x = torch.cat(args, dim=-1)
        if self.observe_alpha:
            v = torch.zeros(*x.shape[:-1], len(self.alphas), device=x.device, dtype=x.dtype) + self.alphas
            x = torch.cat([x, v], dim=-1)
        return x

    def build_backbone(self, inp_dim, z_dim, hidden_dim, output_shape):
        if self.backbone_type == 'mlp' or z_dim == 0:
            net = mlp(inp_dim + self.alpha_dim + self.time_embedding + z_dim, hidden_dim, output_shape)
            return Seq(net)
        else:
            assert self.alpha_dim == 0
            assert self.time_embedding == 0
            from nn.modules.sequence import SequentialBackbone
            return SequentialBackbone(inp_dim, z_dim, output_shape)


def batch_select(values, z=None):
    if z is None:
        return values
    else:
        out = torch.gather(values, -1, z.unsqueeze(-1))
        #print(values[2, 10, z[2, 10]], out[2, 10])
        return out

class SoftQPolicy(BackboneBase):
    def __init__(
        self,state_dim, action_dim, z_space, hidden_dim, cfg = None,
    ) -> None:
        #nn.Module.__init__(self)
        BackboneBase.__init__(self)
        self.z_space = z_space
        self.enc_z = z_space.tokenize
        self.q = self.build_backbone(state_dim + action_dim + z_space.dim, hidden_dim, 1)
        self.q2 = self.build_backbone(state_dim + action_dim + z_space.dim, hidden_dim, 1)
        self.action_dim = action_dim

    from tools.utils import print_input_args

    def forward(self, s, z, a, prevz=None, timestep=None, r=None, done=None, new_s=None, gamma=None):
        raise NotImplementedError("Q is not supported yet ..")
        z = self.enc_z(z)
        if self.action_dim > 0:
            inp = self.add_alpha(s, a, z, timestep=timestep)
        else:
            assert torch.allclose(a, z)
            inp = self.add_alpha(s, z, timestep=timestep)

        q1 = self.q(inp)
        q2 = self.q2(inp)
        q_value = torch.cat((q1, q2), dim=-1)
        return q_value, None

    def get_predict(self, rollout):
        return rollout['q_value']

    def compute_target(self, vtarg, reward, done_gt, gamma):
        assert vtarg.shape == reward.shape == done_gt.shape
        return reward + (1-done_gt.float()) * gamma * vtarg


class ValuePolicy(BackboneBase):
    def __init__(
        self,state_dim, action_dim, z_space, hidden_dim, cfg = None,
        zero_done_value=True,
    ) -> None:
        #nn.Module.__init__(self)
        BackboneBase.__init__(self)
        self.z_space = z_space
        self.enc_z = z_space.tokenize
        # assert isinstance(z_space, spaces.Discrete)
        self.q = self.build_backbone(state_dim, z_space.dim, hidden_dim, 1)
        self.q2 = self.build_backbone(state_dim, z_space.dim, hidden_dim, 1)

    def forward(self, s, z, a, prevz=None, timestep=None, r=None, done=None, new_s=None, gamma=None):
        # return the Q value .. if it's value, return self._cfg.gamma
        z = self.enc_z(z)
        mask = 1. if done is None else (1-done.float())
        # print(new_s.shape, new_s.device, z.shape, z.device)
        #inp = self.add_alpha(new_s, z, timestep=timestep+1)

        v1, v2 = self.q(new_s, z), self.q2(new_s, z)
        values = torch.cat((v1, v2), dim=-1)
        if self._cfg.zero_done_value:
            mask = 1. # ignore the done in the end ..
        q_value = values * gamma * mask + r
        return q_value, values

    def get_predict(self, rollout):
        return rollout['pred_values']

    def compute_target(self, vtarg, reward, done_gt, gamma):
        assert vtarg.shape == done_gt.shape
        if self._cfg.zero_done_value:
            vtarg = vtarg * (1 - done_gt.float())
        return vtarg