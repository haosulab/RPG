import torch
import numpy as np
from tools import Configurable
from tools.optim import LossOptimizer
from tools.utils import weighted_sum_dict, summarize_info

from .envs import GoalEnv
from .utils import reverse_cumsum
from .rpg_net import RPGActor, RPGCritic, InfoNet

from tools.utils import dshape, dstack, detach


# algorithm for training the networks ..
class RPG(Configurable):
    def __init__(
        self,
        actor: RPGActor,
        info_log_q: InfoNet,
        critic: RPGCritic,
        prior_actor,
        cfg=None,

        optim=LossOptimizer.get_default_config(),
        info_log_q_optim=None,
        value_optim=None,

        
        use_action_entropy = False,

        baseline=True,

        weight=dict(
            ent_z=1.,
            ent_a=1.,
            mutual_info=1.,
            reward=1.,
            prior=None
        ),
        gamma=0.97,
        lmbda=0.97,  # RL parameters
        gd=True,
        stop_pg=False,
    ):
        super().__init__()
        if self._cfg.weight.prior is None:
            self._cfg.weight.prior = self._cfg.weight.ent_a

        self.actor = actor
        self.info_log_q = info_log_q
        self.critic = critic
        self.prior_actor = prior_actor

        self.pi_optim = LossOptimizer(
            self.actor, cfg=optim)
        self.critic_optim = LossOptimizer(
            self.critic, cfg=value_optim or optim)
        self.info_q_phi_optim = LossOptimizer(
            self.info_log_q, cfg=info_log_q_optim or optim)
        self.baseline = None
        self.best = -np.inf

    def inference(self, env: GoalEnv, z=None, start_timestep=None,
                  prior=False, label_traj=None, values=None, return_state=False, **kwargs):

        assert values is None, "critic is not supported yet .."

        obs = env.reset(**kwargs)
        data, infos = [], []
        actor = self.actor if not prior else self.prior_actor
        if start_timestep is None:
            start_timestep = 0
        if z is None:
            z = self.prior_actor.pi_z(obs, None, timestep=0).sample()[0] * 0


        if return_state:
            states = []

        for step in range(start_timestep, start_timestep + env.low_steps):
            z_old = z
            p_z = actor.pi_z(obs, z, timestep=step)

            if label_traj is None:
                z, log_p_z = p_z.sample()
            else:
                z = label_traj[0][step]
                log_p_z = p_z.log_prob(z)

            # from .distributions import CategoricalAction
            # log_p_z = log_p_z.detach()

            p_a = actor.pi_a(obs, z, timestep=step)
            if label_traj is None:
                if self._cfg.gd:
                    a, log_p_a = p_a.rsample()
                else:
                    a, log_p_a = p_a.sample()

                if self._cfg.use_action_entropy:
                    log_p_a = - p_a.entropy().sum(axis=-1)
            else:
                a = label_traj[1][step]
                log_p_a = p_a.log_prob(a)

            assert log_p_a.shape == log_p_z.shape

            if return_state:
                states.append(env.get_state())

            next_obs, r, _, info = env.step(a)
            data.append([obs, r, a, log_p_a, z, log_p_z])
            infos.append(info)

            if values is not None:
                values.append(self.critic(
                    obs.detach(), z.detach(), z_old, timestep=step))

            obs = next_obs

        if values is not None:
            values.append(self.critic(obs.detach(), z.detach(),
                          z.detach(), timestep=step+1))

        data = dict(zip(['s', 'r', 'a', 'log_p_a', 'z', 'log_p_z', 'info', 'last_obs'],
                        list(map(lambda x: dstack(x, device=actor.device), zip(*data)))+[infos, actor.batch_input(obs)]))

        if return_state:
            import numpy as np
            data['state'] = dstack(states, device=actor.device)
            data['last_state'] = env.get_state()

        return data

    def elbo(self, s, r, a, z, log_p_a, log_p_z, **trajs):
        # print(z[:, 0])
        prior = self.prior_actor.pi_a([None], None).log_prob(a)
        mutual_info = self.info_log_q(s, a, detach(z))
        elbo, info = weighted_sum_dict(
            dict(reward=r, mutual_info=mutual_info, ent_z=-log_p_z, ent_a=-log_p_a, prior=prior),
            self._cfg.weight
        )
        return elbo, mutual_info, info

    def compute_advantages(self, r, values, **trajs):
        gamma = self._cfg.gamma
        lmbda = self._cfg.lmbda
        if lmbda > 0:
            assert len(values) == len(r) + 1

            #GAE for action ..
            next_step_value = r + gamma * values[1:][..., 0]  # V(s, old_z)
            delta = next_step_value[..., None] - values[:-1]
            adv = torch.zeros_like(delta)
            # USE GAE here ..
            last_gae = 0
            for i in range(len(r)-1, -1, -1):
                last_gae = adv[i] = last_gae + delta[i] * gamma * lmbda
        else:
            r = reverse_cumsum(r) + values[-1][..., 0]
            adv = r - values[:-1]
            raise NotImplementedError("Need to decay the rewards ..")

        return adv, adv + values[:-1]
    # --------------------------------- optimize ---------------------------------------------

    def optimize_info_q(self, s, a, z, **trajs):
        loss = -self.info_log_q(detach(s), detach(a), detach(z))
        loss = loss.sum(axis=0).mean()
        self.info_q_phi_optim.optimize(loss)
        return {'info_q_loss': loss.item()}

    def optimize_critic(self, values, vtargs, **trajs):
        #value = self.critic(s, z)
        assert values.shape == vtargs.shape
        assert values.shape[-1] == 2
        z_loss = ((values[..., 0] - vtargs[..., 0])**2).mean()
        a_loss = ((values[..., 1] - vtargs[..., 1])**2).mean()
        self.critic_optim.optimize(z_loss + a_loss)
        return {'value_z_loss': z_loss.item(), 'value_a_loss': a_loss.item()}

    def direct_optimize(self, traj):
        log_p_z = traj['log_p_z']


        elbo, _, info = self.elbo(**traj)

        accumulated_reward = reverse_cumsum(elbo.detach(), 0)

        with torch.no_grad():
            if self._cfg.baseline:
                if self.baseline is None:
                    self.baseline = 0
                
                self.baseline = self.baseline * 0.9 + accumulated_reward.mean(dim=1, keepdims=True).detach() * 0.1
                accumulated_reward = accumulated_reward - self.baseline

        assert accumulated_reward.shape == log_p_z.shape, f"{elbo.shape}, {log_p_z.shape}"
        policy_gradient = -(log_p_z * accumulated_reward).sum(axis=0).mean()

        if self._cfg.stop_pg:
            policy_gradient = policy_gradient.detach()

        if self._cfg.gd:
            differentiable_part = -elbo.sum(axis=0).mean()
        else:
            differentiable_part = -(traj['log_p_a'] * accumulated_reward).sum(axis=0).mean()

        self.pi_optim.optimize(differentiable_part + policy_gradient)

        if self.actor._cfg.softmax_policy:
            z = traj['z'][0].detach().cpu().numpy()
            r = traj['r'].sum(axis=0)#elbo.detach().sum(axis=0).detach().cpu().numpy()
            for i in range(len(z)):
                self.actor.counter[z[i]] += 1
                self.actor.rewards[z[i]] += r[i]

        return {
            'pg': policy_gradient.item(),
            'gd': differentiable_part.item(),
            'best': traj['r'].sum(axis=0).max().item(),
            **info,
        }

    def run_batch(self, env, info_q_iter=1, **kwargs):
        traj = self.inference(env, **kwargs)
        losses = dict()
        for reduce_method in ["sum", "max", "min"]:
            losses.update(summarize_info(traj['info'], reduce=reduce_method))
        if self._cfg.gd:
            traj["a"].retain_grad()
        losses.update(self.direct_optimize(traj))
        if losses['best'] > self.best:
            self.best = losses['best']
        losses['sofar'] = self.best

        if self._cfg.gd:
            with torch.no_grad():
                a_grad = traj["a"].grad.view(-1, traj['a'].shape[-1])
                a_grad_avg  = a_grad.mean(dim=0)
                a_grad_var   = ((a_grad - a_grad_avg) ** 2).sum(-1).mean()
                a_grad_norms = a_grad.norm(-1)
                a_norm_max   = a_grad_norms.max()
                a_norm_min   = a_grad_norms.min()
                a_norm_avg   = a_grad_norms.mean()
                losses["a_grad_var"] = a_grad_var.item()
                losses["a_norm_max"] = a_norm_max.item()
                losses["a_norm_min"] = a_norm_min.item()
                losses["a_norm_avg"] = a_norm_avg.item()
        
        for i in range(info_q_iter):
            losses.update(self.optimize_info_q(**traj))
        return losses


    # -----------------------  render ----------------------------------------------
    @torch.no_grad()
    def evaluate_elbo(self, env, batch_size, log_p_o=None, **kwargs):
        from .elbo_checker import evaluate_elbo_from_prior, evaluate_log_p_O
        if log_p_o is None:
            log_p_o = evaluate_log_p_O(self, env, batch_size, n_batch=200)
        evaluate_elbo_from_prior(self, env, batch_size=batch_size, log_p_o=log_p_o, **kwargs)
        return log_p_o