from tools.config import Configurable
import einops
import re
import numpy as np
import torch
from .worldmodel import HiddenDynamicNet

def linear_schedule(schdl, step):
    """
    Outputs values following a linear decay schedule.
    Adapted from https://github.com/facebookresearch/drqv2
    """
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
    raise NotImplementedError(schdl)


class CEM(Configurable):
    def __init__(self, 
                worldmodel: HiddenDynamicNet,
                horizon,
                action_dim,
                cfg=None,
                seed_steps=1000,
                num_samples=512,
                iterations=6,
                mixture_coef=0.05,
                min_std = 0.01,
                temperature= 0.5,
                momentum= 0.1,
                num_elites=64,
        ) -> None:
        super().__init__()
        self.cfg = cfg
        self.horizon = horizon
        self.worldmodel: HiddenDynamicNet = worldmodel
        self.action_dim = action_dim
        self.horizon_schedule = f'linear(1, {horizon}, 25000)'


    def plan(self, obs, timesteps, z, eval_mode=False, step=None, **kwargs):
        # copied from tdmpc and make it support batch version.
        # Seed steps
        from tools.utils import totensor

        action_dim = self.action_dim
        device = 'cuda:0'
        if step < self.cfg.seed_steps and not eval_mode:
            return torch.empty(obs.shape[0], self.action_dim, dtype=torch.float32, device=device).uniform_(-1, 1)

        # Sample policy trajectories
        #obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        #horizon = int(min(self.horizon, linear_schedule(self.horizon_schedule, step)))
        horizon = self.horizon
        num_pi_trajs = int(self.cfg.mixture_coef * self.cfg.num_samples)

        batch_size = obs.shape[0]
        if num_pi_trajs > 0:
            _obs, _z, _t = [einops.repeat(x, 'b ... -> (n b) ...', n=num_pi_trajs) for x in [obs, z, timesteps]]
            data = self.worldmodel.inference(_obs, _z, _t, step=horizon, **kwargs)
            pi_actions = einops.rearrange(data['a'], 't (n b) ... -> t n b ...', n=num_pi_trajs) # ((b n) n)

        if self.cfg.iterations == 0:
            return pi_actions[0, 0]
        # Initialize state and parameters
        device = pi_actions.device
        mean = torch.zeros(horizon, batch_size, action_dim, device=device)
        std = 2*torch.ones(horizon, batch_size, action_dim, device=device)

        t0 = (timesteps == 0)
        if hasattr(self, '_prev_mean') and self._prev_mean is not None and t0.any():
            mean[:, t0, :-1] = self._prev_mean[:, t0, 1:]

        # Iterate CEM
        total_samples = self.cfg.num_samples + num_pi_trajs
        old_obs = obs
        obs, z, timesteps = [einops.repeat(x, 'b ... -> (n b) ...', n=total_samples, b=batch_size) 
                       for x in [obs, z, timesteps]]

        for i in range(self.cfg.iterations):
            actions = torch.clamp(mean.unsqueeze(1) + std.unsqueeze(1) * \
                torch.randn(horizon, self.cfg.num_samples, batch_size, action_dim, device=std.device), -1, 1)

            if num_pi_trajs > 0:
                actions = torch.cat([actions, pi_actions], dim=1) # T, n, b, action_dim

            a_seq = einops.rearrange(actions, 't n b ... -> t (n b)  ...', b=batch_size, t=horizon, n=total_samples)
            value = self.worldmodel.inference(obs, z, timesteps, horizon, a_seq=a_seq, **kwargs)['value']
            value = einops.rearrange(value, '(n b) ... -> n b ...', b=batch_size, n=total_samples)

            elite_idxs = torch.topk(value[..., 0], self.cfg.num_elites, dim=0).indices # K, b
            elite_value = torch.gather(value, 0, elite_idxs.unsqueeze(-1)) # K, b, 1
            elite_actions = torch.gather(actions, 1, elite_idxs.unsqueeze(0).unsqueeze(-1).expand(horizon, -1, -1, action_dim))

            #assert torch.allclose(elite_actions[:, 10, 2], actions[:, elite_idxs[10, 2], 2])
            # T, K, b, action_dim
            #print(elite_value.shape, elite_actions.shape)

            # elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

            # Update parameters
            max_value = elite_value.max(0)[0] # b, 1
            score = torch.exp(self.cfg.temperature*(elite_value - max_value))
            score /= score.sum(0, keepdim=True) # K, b, 1
            #print(score.shape, score_sum.shape, elite_actions.shape)
            _mean = torch.sum(score[None, :] * elite_actions, dim=1) / (score.sum(0, keepdim=True) + 1e-9)
            _std = torch.sqrt(torch.sum(score[None,:] * (elite_actions - _mean[:, None,:]) ** 2, dim=1) / (score.sum(0, keepdim=True) + 1e-9))
            _std = _std.clamp_(self._cfg.min_std, 2) #TODO: change std
            mean, std = self.cfg.momentum * mean + (1 - self.cfg.momentum) * _mean, _std

        # Outputs
        # score = score.squeeze(1).cpu().numpy()
        #score = score.detach().cpu().numpy()
        score = score[..., 0].permute(1, 0)
        index = torch.distributions.Categorical(score).sample()
        index = index[None, None, :, None].expand(horizon, -1, -1, action_dim)
        actions = torch.gather(actions, 1, index)[:, 0]

        self._prev_mean = mean
        mean, std = actions[0], _std[0]
        a = mean
        if not eval_mode:
            a += std * torch.randn(action_dim, device=std.device)
        return a


if __name__ == '__main__':
    class FakeInference:
        def __init__(self, action_dim):
            self.action_dim = action_dim

        def inference(self, obs, z, timesteps, step, a_seq=None, **kwargs):
            
            if a_seq is None:
                a_seq = torch.randn(step, len(obs), self.action_dim).cuda().clamp(-1., 1.)
            assert len(a_seq) == step
            #print('within inference', a_seq.sum(axis=0)[0], obs[0])
            value = -((a_seq.sum(axis=0) - obs)**2).sum(axis=-1, keepdim=True)
            
            #print(value[0])
            return {'a': a_seq, 'value': value}

    inference = FakeInference(2)
    cem = CEM(inference, horizon=4, action_dim=2)
    

    obs = torch.tensor(
        [[1., 0.], [0., 1.], [-1., 0.], [1., 0.]],
    ).cuda()
    print(cem.plan(obs, torch.zeros(4).long(), torch.zeros(4).float(), step=50000))