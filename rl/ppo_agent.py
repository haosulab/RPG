import torch
from torch import nn
import tqdm
import numpy as np
from .models.backbone import batch_input
from tools.config import merge_inputs
from .models import Value
from .agent import Agent
from .optims import PPO as PPOOptim, ValueOptim, AuxOptim
from .roller import roller, RollingState
from .vec_envs import SubprocVectorEnv
from tools.utils import logger


def compute_gae(
    vpred,
    next_vpred,
    reward,
    done,
    gamma,
    lmbda,
    _norm_adv,
    _epsilon,
    rew_rms,
):
    if isinstance(vpred, list):
        vpred = np.array(vpred)
        reward = np.array(reward)

    if rew_rms is not None:
        vpred = vpred * rew_rms.std
        next_vpred = next_vpred * rew_rms.std

    nstep = len(vpred)
    if isinstance(vpred[0], np.ndarray):
        adv = np.zeros((nstep, len(vpred[0])), np.float32)
    else:
        adv = torch.zeros((nstep, len(vpred[0])), device=vpred.device, dtype=vpred.dtype)
    lastgaelam = 0

    for t in reversed(range(nstep)):
        nextvalue = next_vpred[t]
        mask = 1. - done[t]
        assert mask.shape == nextvalue.shape
        #print(reward.device, next_vpred.device, mask.device, vpred[t].device)
        delta = reward[t] + gamma * nextvalue * mask - vpred[t]
        adv[t] = lastgaelam = delta + gamma * lmbda * lastgaelam * mask
        # nextvalue = vpred[t]

    vtarg = vpred + adv
    #reward = reward

    if rew_rms is not None:
        rew_rms.update(vtarg.reshape(-1))
        # https://github.com/haosulab/pyrl/blob/926d3d07d45f3bf014e7c6ea64e1bba1d4f35f03/pyrl/utils/torch/module_utils.py#L192
        vtarg = vtarg / rew_rms.std

    logger.logkvs({
        'reward_mean': reward.mean(),
        'reward_std': reward.std(),
        'vtarg_mean': vtarg.mean(),
        'orig_vpred_mean': vpred.mean(),
        'orig_vpred_std': vpred.std(),  # NOTE: this is denormalized ..
    })

    if _norm_adv:
        adv = (adv - adv.mean()) / (adv.std() + _epsilon)
        logger.logkvs({
            'adv_normed_mean': adv.mean(),
            'adv_normed_std': adv.std(),
        })

    return vtarg, adv


def sample_batch(index, batch_size):
    return np.array_split(np.array(index), max(len(index)//batch_size, 1))


def minibatch_gen(traj, index, batch_size, KEY_LIST=None, verbose=False):
    if KEY_LIST is None:
        KEY_LIST = list(traj.keys())

    tmp = sample_batch(np.random.permutation(len(index)), batch_size)
    if verbose:
        tmp = tqdm.tqdm(tmp, total=len(tmp))
    for idx in tmp:
        k = index[idx]
        yield {
            key: [traj[key][i][j] for i, j in k]
            for key in KEY_LIST
        }


class PPOAgent(Agent):
    # https://github.com/haosulab/pyrl/blob/master/configs/mfrl/ppo/mujoco.py
    # https://github.com/haosulab/pyrl/blob/master/pyrl/methods/mfrl/ppo.py

    KEY_LIST = ['obs', 'action', 'logp', 'adv', 'vtarg', 'entropy']

    def __init__(self,
                 obs_space,
                 action_space,
                 cfg=None,

                 ppo_optim=PPOOptim.get_default_config(),
                 critic_optim=ValueOptim.get_default_config(),

                 nsteps=50,
                 roll_times=1,

                 gamma=0.995, lmbda=0.97,
                 adv_norm=True,

                 epsilon=1e-8,

                 batch_size=2000,
                 n_epochs=10,
                 show_roller_progress=False,
                 ignore_episode_done=True,

                 aux_optim=AuxOptim.get_default_config(),
                 ):

        super(PPOAgent, self).__init__(
            obs_space,
            action_space,
            auxiliary=int(aux_optim.mode != None)
        )

        # if soft_ppo:
        #    assert cfg.actor.head.std_mode == 'statewise'

        self.nsteps = nsteps

        kwargs = {'backbone': cfg.actor.backbone}

        # ensure that value has the same backbone as critic..
        self.critic = Value(
            obs_space, **kwargs)

        self.ppo_optim = PPOOptim(
            self.actor, cfg=ppo_optim)

        self.value_optim = ValueOptim(
            self.critic, cfg=critic_optim)

        if aux_optim.mode is not None:
            self.aux_optim = AuxOptim(
                self.actor,
                cfg=aux_optim
            )
            self.aux_buffer = []
        else:
            self.aux_optim = None

        self.rolling_state: RollingState = None
        self.batch_size = batch_size

        self.total_steps = 0

        self.training = False
        self.training_epoch = 0

    def collect_data(self, env: SubprocVectorEnv, **kwargs):
        cfg = merge_inputs(self._cfg, **kwargs)

        ran = tqdm.trange if cfg.roll_times > 1 else range
        outs = []
        for _ in ran(cfg.roll_times):
            trajectories, self.rolling_state = roller(
                env,
                self.rolling_state,
                self.actor,

                nsteps=cfg.nsteps,

                process_obs=self.process_obs,

                ignore_episode_done=cfg.ignore_episode_done,
                verbose=cfg.show_roller_progress and cfg.roll_times == 1,
            )
            outs += trajectories
        # return trajectories
        return outs

    def predict_value(self, input, index, vpred=None, batch_size=None):
        if batch_size is None:
            batch_size = self._cfg.batch_size

        timesteps = len(input)
        nenv = len(input[0])

        if vpred is None:
            vpred = np.zeros((timesteps, nenv), dtype=np.float32)

        for ind in sample_batch(index, batch_size):
            obs = [input[i][j] for i, j in ind]
            value = self.critic(obs)[:, 0].detach().cpu().numpy()
            for (i, j), v in zip(ind, value):
                vpred[i, j] = v
        return vpred

    def preprocess(self, traj, **kwargs):
        cfg = merge_inputs(self._cfg, **kwargs)

        traj = {key: [i[key] for i in traj if key in i] for key in traj[0]}

        timesteps = len(traj['obs'])
        nenv = len(traj['obs'][0])
        for i in traj:
            assert len(traj[i]) == timesteps

        # compute values ..
        with torch.no_grad():

            vpred = self.predict_value(
                traj['obs'],
                [(i, j) for j in range(nenv) for i in range(timesteps)],
                batch_size=cfg.batch_size
            )

            next_vpred = np.zeros_like(vpred)
            next_vpred[:-1] = vpred[1:]

            # compute next_vpred for episode done location and the last state ..
            next_vpred = self.predict_value(
                traj['next_obs'],
                [(i, j) for j in range(nenv) for i in range(timesteps)
                    if traj['episode_done'] or j == timesteps-1],
                next_vpred,
                batch_size=cfg.batch_size
            )
            # next_vpred_v2 = predict(traj['next_obs'], [(i, j) for j in range(nenv) for i in range(timesteps)])
            # assert np.allclose(next_vpred, next_vpred_v2)

        vtarg, adv = compute_gae(
            vpred=vpred,
            next_vpred=next_vpred,
            reward=np.array(traj['reward']),
            done=np.array(traj['done']),
            gamma=cfg.gamma,
            lmbda=cfg.lmbda,
            _norm_adv=cfg.adv_norm,
            _epsilon=cfg.epsilon,
            rew_rms=self._rew_rms
        )

        traj.update({
            'vtarg': vtarg,
            'adv': adv,
            'timesteps': timesteps,
            'nenv': nenv,
        })
        return traj

    def minibatch_gen(self, traj, KEY_LIST, verbose=False, **kwargs):
        # we use for loop here; the loop is O(n)
        cfg = merge_inputs(self._cfg, **kwargs)

        timesteps, nenv = traj['timesteps'], traj['nenv']
        index = np.array([(i, j) for j in range(nenv)
                          for i in range(timesteps)])
        yield from minibatch_gen(traj, index, cfg.batch_size, KEY_LIST, verbose)

    def train(self, envs, test_envs=None, **kwargs):
        #if self.roller is None:
        #    self.roller = Roller(buffer.roller_env, self.actor, self.critic, cfg=self.roller_cfg)
        self.train_envs = envs
        cfg = merge_inputs(self._cfg, **kwargs)

        self.training = True
        with torch.no_grad():
            raw_data = self.collect_data(envs, **kwargs)
            traj = self.preprocess(raw_data, **kwargs)

        for _ in tqdm.trange(cfg.n_epochs):
            for batch in self.minibatch_gen(traj, self.KEY_LIST):
                obs, action, logp, adv, vtarg, ent = [
                    batch[key] for key in self.KEY_LIST]
                _, val_info = self.value_optim.step(obs, vtarg)
                logger.logkvs_mean(val_info)
                _, optim_info = self.ppo_optim.step(obs, action, logp, adv)
                logger.logkvs_mean(optim_info)
                if 'early_stop' in optim_info and optim_info['early_stop']:
                    break

        self.training = False

        self.training_epoch += 1
        if self.save_episode and self.training_epoch % self.save_episode == 0:
            assert test_envs is None, "not unit tested yet"
            self.eval_and_save(test_envs or envs)

        self.total_steps += traj['timesteps'] * traj['nenv']
        logger.logkv('total_env_steps', self.total_steps)
        logger.logkv('epoch', self.training_epoch)

        if self.aux_optim is not None:
            # drop other data for saving memory
            self.aux_buffer.append([{'obs': i['obs']} for i in raw_data])
            if len(self.aux_buffer) == self.aux_optim._cfg.ppo_epoch:
                self.aux_optim.train(
                    self.aux_buffer,
                    cfg.batch_size,
                    self.predict_value,
                    self._rew_rms
                )
                self.aux_buffer = []

        logger.dumpkvs()
