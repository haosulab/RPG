# test the soft q ..
from .soft_rpg_old import *
from .critic import Aout

class IdentityPolicy(torch.nn.Module):
    def forward(self, state_embed, hidden):
        return Aout(hidden, torch.zeros(len(state_embed), device=state_embed.device))

    def loss(self, rollout):
        return -rollout['value'][..., 0].mean() * 0.

    def set_alpha(self, *args, **kwargs):
        pass

class QTrainer(Trainer):
    def __init__(self, env, cfg=None):
        super().__init__(env, cfg)

    def make_network(self, obs_space, action_space, z_space):
        z_space = self.z_space = action_space

        hidden_dim = 256
        state_dim = 100
        enc_s, enc_z, enc_a, init_h, dynamics, reward_predictor, done_fn, state_dec = self.make_dynamic_network(
            obs_space, action_space, z_space, hidden_dim, state_dim)

        if self._cfg.qmode == 'Q':
            q_fn = SoftQPolicy(state_dim, 0, z_space, hidden_dim)
        else:
            q_fn = ValuePolicy(state_dim, 0, z_space, enc_z, hidden_dim)

        pi_z = SoftPolicyZ(state_dim, hidden_dim, enc_z, cfg=self._cfg.pi_z, K=1) # select K every steps
        pi_a = IdentityPolicy()

        network = GeneralizedQ(
            enc_s, enc_a, pi_a, pi_z,
            init_h, dynamics, state_dec, reward_predictor, q_fn,
            done_fn, None,
            gamma=self._cfg.gamma,
            lmbda=self._cfg.lmbda,
            horizon=self._cfg.horizon
        )

        network.apply(orthogonal_init)

        info_net = self.make_intrinsic_reward(
            obs_space, action_space, z_space, hidden_dim, state_dim
        )
        return network.cuda(), info_net