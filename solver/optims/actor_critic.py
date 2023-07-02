import torch
from tools.optim import OptimModule 
from tools.utils import summarize_info, batch_input

"""
class PolicyOptim(OptimModule):
    name = 'actor'
    KEYS = ['state', 'goal']

    def __init__(self, policy, roller, cfg=None,
                 lr=0.0005, max_grad_norm=0.5, 
                 gd=1., pg=0., # gradient descent, policy gradient
    ):
        super(PolicyOptim, self).__init__(policy)
        from ..diff_agent import Roller
        self.roller: "Roller" = roller

    def forward_network(self, obs, **kwargs):
        return super().forward_network(obs, **kwargs)

    def _compute_loss(self, state, goal, backward, update_buffer=False):
        # if we need to compute value
        batch_infos = []
        num_state = len(state)
        total_loss = 0.
        for s, g in zip(state, goal):
            info = self.roller.rollout(self, s, g, update=update_buffer)

            traj = info.pop('traj')
            loss = info.pop('loss'); assert loss.shape == (len(s),), f"{loss.shape}"
            loss = loss.mean() * self._cfg.gd # differentiable physics loss

            if update_buffer or self._cfg.pg > 0.:
                with torch.no_grad():
                    advs = self.roller.process_trajs(traj, update=update_buffer)

            if self._cfg.pg > 0.:
                advs = torch.tensor(advs, device=self.device, dtype=torch.float32).sum(axis=-1)
                logp = torch.stack(traj['logp'])
                assert logp.shape  == advs.shape, f"{logp.shape}, {advs.shape}"

                pg = (logp * advs).mean(axis=1).sum(axis=0)
                info['pg_loss'] = pg.item()
                info['advs'] = advs.mean().item()
                loss -= pg * self._cfg.pg  # maximize the lopg times adv

            if backward:
                (loss/num_state).backward()
            total_loss += loss.item()
            info['loss'] = loss.item()
            info['reward'] = traj['reward'].sum(axis=0).mean().item()
            batch_infos.append(info)
        batch_infos = summarize_info(batch_infos, 'mean')
        return total_loss, batch_infos


class ValueOptim(OptimModule):
    KEYS = ['obs', 'vtarg']
    name = 'value'

    def __init__(self, network, cfg=None, max_grad_norm=0.5):
        super().__init__(network, cfg)

    def forward_network(self, obs):
        return self.network(obs)

    def compute_loss(self, obs, vtarg):
        return super().compute_loss(obs, batch_input(vtarg, self.device))
"""

class ValueOptim(OptimModule):
    def __init__(self, critic, cfg=None,
                 lr=5e-4, vfcoef=0.5):
        super(ValueOptim, self).__init__(critic)
        self.critic = critic
        #self.optim = make_optim(critic.parameters(), lr)
        self.vfcoef = vfcoef

    def compute_loss(self, obs, vtarg):
        vpred = self.critic(obs).mean[..., 0]
        vtarg = batch_input(vtarg, vpred.device)
        assert vpred.shape == vtarg.shape
        vf = self.vfcoef * ((vpred - vtarg) ** 2).mean()
        return vf


class PPO(OptimModule):
    # no need to store distribution, we only need to store actions ..
    def __init__(self,
                 actor,
                 cfg=None,
                 lr=5e-4,
                 clip_param=0.2,
                 max_kl=None,
                 max_grad_norm=None,
                 ):
        super(PPO, self).__init__(actor)
        self.actor = actor
        self.clip_param = clip_param

    def _compute_loss(self, obs, action, logp, adv, backward=True):
        from ..distributions import ActionDistr
        pd: ActionDistr = self.actor(obs)

        newlogp = pd.log_prob(action)
        device = newlogp.device

        adv = batch_input(adv, device)
        logp = batch_input(logp, device)

        # prob ratio for KL / clipping based on a (possibly) recomputed logp
        logratio = newlogp - logp
        ratio = torch.exp(logratio)
        assert newlogp.shape == logp.shape
        assert adv.shape == ratio.shape, f"Adv shape is {adv.shape}, and ratio shape is {ratio.shape}"

        if self.clip_param > 0:
            pg_losses = -adv * ratio
            pg_losses2 = -adv * \
                torch.clamp(ratio, 1.0 - self.clip_param,
                            1.0 + self.clip_param)
            pg_losses = torch.max(pg_losses, pg_losses2)
        else:
            raise NotImplementedError

        assert len(pg_losses.shape) == 2, f"{pg_losses.shape}"
        pg_losses = pg_losses.sum(axis=0).mean()
        #loss = pg_losses

        #approx_kl_div = (ratio - 1 - logratio).mean().item()
        #early_stop = self._cfg.max_kl is not None and (
        #    approx_kl_div > self._cfg.max_kl * 1.5)
        early_stop = False
        assert self._cfg.max_kl is None

        if not early_stop and backward:
            pg_losses.backward()
        else:
            pass

        output = {
            'pg': pg_losses.item(),
            #'approx_kl': approx_kl_div,
        }
        if self._cfg.max_kl:
            output['early_stop'] = early_stop
        return pg_losses, output
