import torch
import numpy as np
from torch import nn
import torch.optim
from tools.config import Configurable, as_builder
import torch as th
from .models import ActionDistr
from tools.utils import logger, batch_input

from tools.optim import OptimModule as Optim


class PPO(Optim):
    # no need to store distribution, we only need to store actions ..
    def __init__(self,
                 actor,
                 cfg=None,
                 lr=5e-4,
                 clip_param=0.2,
                 entropy_coef=0.0,
                 max_kl=None,
                 max_grad_norm=0.5,
                 mode='step',
                 ):
        super(PPO, self).__init__(actor)

        self.actor = actor
        self.entropy_coef = entropy_coef
        self.clip_param = clip_param

    def _compute_loss(self, obs, action, logp, adv, backward=True):
        pd: ActionDistr = self.actor(obs)

        newlogp = pd.log_prob(action)
        device = newlogp.device

        adv = batch_input(adv, device)
        logp = batch_input(logp, device)

        # prob ratio for KL / clipping based on a (possibly) recomputed logp
        logratio = newlogp - logp
        ratio = th.exp(logratio)
        assert newlogp.shape == logp.shape
        assert adv.shape == ratio.shape, f"Adv shape is {adv.shape}, and ratio shape is {ratio.shape}"

        if self.clip_param > 0:
            pg_losses = -adv * ratio
            pg_losses2 = -adv * \
                th.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            pg_losses = th.max(pg_losses, pg_losses2)
        else:
            raise NotImplementedError

        entropy = pd.entropy(sum=True)
        assert entropy.shape == pg_losses.shape

        pg_losses, entropy = pg_losses.mean(), entropy.mean()
        negent = -entropy * self.entropy_coef

        loss = negent + pg_losses

        approx_kl_div = (ratio - 1 - logratio).mean().item()
        early_stop = self._cfg.max_kl is not None and (
            approx_kl_div > self._cfg.max_kl * 1.5)

        if not early_stop and backward:
            loss.backward()
        else:
            pass

        output = {
            'entropy': entropy.item(),
            'negent': negent.item(),
            'pi': loss.item(),
            'pg': pg_losses.item(),
            'approx_kl': approx_kl_div,
        }
        if self._cfg.max_kl:
            output['early_stop'] = early_stop
        return loss, output


class ValueOptim(Optim):
    def __init__(self, critic, cfg=None,
                 lr=5e-4, vfcoef=0.5, mode='step'):
        super(ValueOptim, self).__init__(critic)
        self.critic = critic
        #self.optim = make_optim(critic.parameters(), lr)
        self.vfcoef = vfcoef

    def compute_loss(self, obs, vtarg):
        vpred = self.critic(obs)[..., 0]
        vtarg = batch_input(vtarg, vpred.device)
        assert vpred.shape == vtarg.shape
        vf = self.vfcoef * ((vpred - vtarg) ** 2).mean()
        return vf



class AuxOptim(Optim):
    # https://github.com/hzaskywalker/PlasticineLabV2/blob/20f190700a941115fbc9e47af9e76d7c571a0481/plb/algorithms/ppg/train.py
    def __init__(
        self,
        actor,
        cfg=None,
        lr=5e-4,
        batch_size=None,  # by default, use the same batch size as the others
        n_epoch=6,  # should after 6 epoch of actor training
        mode=None,
        beta_clone=1.,
        aux_weight=1.,
        ppo_epoch=32,  # optimized after 32 ppo epoch
    ) -> None:
        super().__init__(actor)
        self.actor = actor
        assert mode in ['value', 'reward']

    @torch.no_grad()
    def prepare_aux(
        self,
        trajs,
        batch_size,
        value_predictor=None,
        rew_rms=None,
    ):
        print('start auxiliary..')

        # concatenate trajectories together
        traj = sum(trajs, [])
        timesteps, nenv = len(traj), len(traj[0]['obs'])

        traj = {key: [i[key] for i in traj if key in i] for key in traj[0]}
        index = np.array([(i, j) for j in range(nenv) for i in range(timesteps)])

        if self._cfg.mode == 'value':
            assert value_predictor is not None
            aux = value_predictor(
                traj['obs'],
                index,
                self._cfg.batch_size
            )
        else:
            aux = np.array(trajs['reward'])
            if rew_rms is not None:
                aux = aux / rew_rms.std  # normalize the rewards

        from .ppo_agent import sample_batch  # TODO: put this to the utils

        #TODO: refactor this..
        pd = [[None for j in range(nenv)] for i in range(timesteps)]
        input = traj['obs']
        for ind in sample_batch(index, batch_size):
            obs = [input[i][j] for i, j in ind]
            for (i, j), v in zip(ind, self.actor(obs, aux=False).to('cpu')):
                pd[i][j] = v

        return {
            'obs': input,
            'index': index,
            'aux': aux[..., None],
            'pd': pd,
        }

    def compute_loss(self, obs, pd, aux):

        new_pd, new_aux = self.actor(obs, aux=True)
        pd_old = pd[0].stack(pd).to('cuda:0')
        assert pd_old.mean.shape == new_pd.mean.shape
        pol_distance = torch.distributions.kl_divergence(
            pd_old.dist, new_pd.dist).mean() * self._cfg.beta_clone

        aux_gt = batch_input(aux, device='cuda:0')
        assert new_aux.shape == aux_gt.shape, f"{new_aux.shape}, {aux_gt.shape}"
        aux_loss = ((aux_gt - new_aux) ** 2).mean() * self._cfg.aux_weight
        return aux_loss + pol_distance, {
                    'aux_mean': float(aux.mean()),
                    'pol_distance': float(pol_distance),
                    'aux_loss': float(aux_loss),
                }

    def train(self, trajs, batch_size=None, value_predictor=None, rew_rms=None):
        import tqdm
        from .ppo_agent import minibatch_gen
        batch_size = self._cfg.batch_size or batch_size

        data = self.prepare_aux(trajs, batch_size, value_predictor, rew_rms)

        ind = data.pop('index')
        for _ in tqdm.trange(self._cfg.n_epoch):
            for batch in minibatch_gen(data, index=ind, batch_size=batch_size):
                out = self.step(**batch)
                logger.logkvs_mean(out)
