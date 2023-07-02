import copy
import tqdm
import numpy as np
from typing import Optional, Any
from .vec_envs import SubprocVectorEnv
from .models import Actor, Value
from tools.utils import logger
from .models.action import ActionDistr

    
class RollingState:
    # Currently we do not normalize the reward
    def __init__(self, env_num=1):
        self.obs = None
        self.returns = np.zeros(env_num)
        self.lengths = np.zeros(env_num)
        # self.discounted_backward_returns = np.zeros(env_num)


def roller(
    envs: SubprocVectorEnv,
    roll_state: Optional[Any],
    actor: Actor,
    nsteps=0,
    verbose=False,

    ignore_episode_done=True,

    process_obs=lambda x:x,

    training_func=None,
):

    reset_kwargs = {}

    if roll_state is None:
        roll_state = RollingState(envs.env_num)
        obs = envs.reset(**reset_kwargs)
        roll_state.obs = process_obs(obs)


    traj = []

    episode_returns = []
    episode_lengths = []

    ran = tqdm.trange if verbose else range


    if nsteps is None:
        nsteps = 10000000 # INF
        stop_at_done = True
    else:
        stop_at_done = False

    for _ in ran(nsteps):
        # we do not pass gradient to 
        action_distr = actor(roll_state.obs)

        if isinstance(action_distr, ActionDistr):
            actions, logp = action_distr.sample()
            actions = actions.detach().cpu().numpy() 
            entropy = action_distr.entropy().detach().cpu().numpy()
        else:
            actions = action_distr
            logp = entropy = None

        next_obs, reward, done, info = envs.step(actions)
        next_obs = process_obs(next_obs)

        roll_state.returns = roll_state.returns + np.array(reward)
        roll_state.lengths += 1

        transitions = {
            'obs': roll_state.obs,
            'action': actions,
            'logp': logp,
            'reward': reward,
            'entropy': entropy,
            'done': [d and (('TimeLimit.truncated' not in i) or not ignore_episode_done)
                for i, d in zip(info, done)], # when ignore episode done is True, this done will not be False even if the environemt done is met
            'episode_done': done, # terminate by time
            'next_obs': next_obs,
        }
        if logp is not None:
            transitions['logp'] = logp
            transitions['entropy'] = entropy

        if training_func is None:
            traj.append(transitions)
        else:
            training_func(**transitions)

        roll_state.obs = next_obs


        end_envs = np.where(done)[0]

        if len(end_envs) > 0:
            for j in end_envs:
                episode_returns.append(float(roll_state.returns[j]))
                episode_lengths.append(int(roll_state.lengths[j]))
                roll_state.lengths[j] = 0
                roll_state.returns[j] = 0

            for idx, k in zip(end_envs, process_obs(envs.reset(end_envs, **reset_kwargs))):
                roll_state.obs[idx] = k

        if stop_at_done and done[0]:
            break

    if len(episode_returns) > 0:
        outputs = {
            "num_episode": len(episode_returns),
            "rewards": np.mean(episode_returns),
            "avg_len": np.mean(episode_lengths)
        }
        logger.logkvs_mean(outputs)

    return traj, roll_state