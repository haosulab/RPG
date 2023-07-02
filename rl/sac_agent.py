import copy
import numpy as np
import torch
from gym.spaces import Box
from torch import nn
import torch
from torch import nn
from gym.spaces import Box, Discrete
from tools.optim import OptimModule
from .agent import Agent, Actor
from .models import DoubleCritic
# from utils import logger

from .roller import RollingState, roller

from tools import merge_inputs, CN
from tools.utils import RunningMeanStd, logger, batch_input, get_rank
from collections import deque


class Optim(OptimModule):
    def __init__(self, network, cfg=None):
        super().__init__(network)

    def optimize(self, loss):
        self.zero_grad()
        loss.backward()
        self.optim_step()

class SACAgent(Agent):
    def __init__(
        self,
        obs_space: Box,
        action_space: Box,
        cfg: CN = None,
        actor=Actor.get_default_config(
            head=dict(std_mode="statewise", squash=True)),
        actor_optim=Optim.get_default_config(),
        critic=None,
        critic_optim=Optim.get_default_config(),
        entropy_lr=0.0003,

        discount=0.99,
        tau=0.005,

        policy_freq=2,
        target_entropy_coef=1.,

        # normalization in SAC, not sure if it is useful
        obs_norm=False,
        reward_norm=False,
        obs_norm_clip=10.,

        nsteps=None,

        memory_size=int(1e6),
        batch_size=256,
        start_step=25000,
    ):
        Agent.__init__(self, obs_space, action_space, cfg)

        cfg_critic = critic
        if cfg_critic is None:
            cfg_critic = dict(backbone=self._cfg.actor.backbone)

        self.critic = DoubleCritic(
            obs_space,
            action_space,
            cfg=cfg_critic
        )

        self.low = nn.Parameter(
            torch.tensor(action_space.low, dtype=torch.float32), requires_grad=False)
        self.high = nn.Parameter(
            torch.tensor(action_space.high, dtype=torch.float32), requires_grad=False)

        self.actor_optim = Optim(self.actor, cfg=actor_optim)
        self.critic_optim = Optim(self.critic, cfg=critic_optim)

        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.eval()
        self.action_dim = int(np.prod(self.action_space.shape))

        # Target entropy is -|A|.
        self.target_entropy = -self.action_dim * target_entropy_coef

        # We optimize log(alpha), instead of alpha.
        self.log_alpha = nn.Parameter(torch.zeros(1, requires_grad=True))
        self.alpha_optim = Optim(self.log_alpha, lr=entropy_lr, max_grad_norm=None)

        self.rolling_state = None
        self.buffer = None

        self.training_epoch = 0
        self.learning_steps = 0
        self.num_samples = 0


        self.buffer = deque(maxlen=memory_size) # to support arbitrary policies ..


    def learn(self, state, action, reward, next_state, done, cfg):
        self.learning_steps += 1
        device = self.critic.device

        if self._rew_rms is not None:
            reward = reward/self._rew_rms.std # normalize reward here

        state = self.actor.backbone.batch_input(state)
        action = batch_input(action, device)

        reward = batch_input(reward, device).reshape(-1, 1)
        next_state = self.actor.backbone.batch_input(next_state)
        done = batch_input(done, device)

        #raise NotImplementedError("Not squashed action..")
        # Update policy.
        actor_output = self.actor(state)
        sampled_actions, logp = actor_output.rsample()
        entropy = actor_output.entropy()[:, None]
        entropy_term = -logp[:, None] #.sum(dim=-1, keepdim=True) #actor_output.log_prob(sampled_actions, sum=True)[:, None]

        # Expectations of Q with clipped double Q technique.
        qs1, qs2 = self.critic(state, sampled_actions)
        qs = torch.min(qs1, qs2)

        # Policy objective is maximization of (Q + alpha * entropy).
        assert qs.shape == entropy_term.shape, f"{qs.shape}, {entropy_term.shape}"

        # normalize the entropy in consistent with the target q
        policy_loss = -(qs + self.log_alpha.detach().exp() * entropy_term).mean()
        self.actor_optim.optimize(policy_loss)

        # Update the entropy coefficient.
        entropy_loss = -torch.mean(
            self.log_alpha.exp() * (self.target_entropy - entropy_term.detach()).detach())
        self.alpha_optim.optimize(entropy_loss)

        # Calculate current and target Q values.
        qs1, qs2 = self.critic(state, action)
        with torch.no_grad():
            actor_output = self.actor(next_state)
            next_action, next_log_prob = actor_output.sample()
            next_log_prob = next_log_prob[:, None]

            next_qs1, next_qs2 = self.critic_target(next_state, next_action)
            next_qs = torch.min(next_qs1, next_qs2) 

            assert next_qs.shape == next_log_prob.shape, f"{next_qs.shape}, {next_log_prob.shape}"
            next_qs = next_qs - self.log_alpha.detach().exp() * next_log_prob

            assert reward.shape == next_qs.shape, f"{reward.shape}, {next_qs.shape}"
            target_qs = reward + (1.0 - done) * cfg.discount * next_qs

        # Q loss is mean squared TD errors with importance weights.
        q1_loss = torch.nn.functional.mse_loss(qs1, target_qs)
        q2_loss = torch.nn.functional.mse_loss(qs2, target_qs)
        q_loss = q1_loss + q2_loss
        self.critic_optim.optimize(q1_loss + q2_loss)

        # Return three values for DisCor algorithm.
        with torch.no_grad():
            logger.logkvs_mean(
                {
                    'loss/policy': policy_loss.mean().item(),
                    'loss/entropy': entropy_loss.mean().item(),
                    'loss/q': q_loss.mean().item(),
                    'stats/alpha': self.log_alpha.exp().item(),
                    'stats/entropy': entropy.mean().item(),
                }
            )
            if self._rew_rms is not None:
                logger.logkvs_mean({
                    'reward_norm': self._rew_rms.std
                })

            if self.learning_steps % cfg.policy_freq == 0:
                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(
                        cfg.tau * param.data + (1 - cfg.tau) * target_param.data)

    
    def train(self, envs, test_envs=None, **kwargs):
        # this is a roller similar to the ppo one 
        # if nsteps is None, that means we terminate at the end of the episode
        cfg = merge_inputs(self._cfg, **kwargs)

        self.training_epoch += 1

        def start_training():
            return len(self.buffer) >= cfg.start_step

        def training_func(obs, action, reward, done, next_obs, **kwargs):
            self.num_samples += len(obs)
            if self._rew_rms is not None:
                self._rew_rms.update(reward)

            for o, a, r, d, next in zip(obs,action, reward, done, next_obs):
                #self.buffer.append(o, a, r, next, d, real_done=None)
                self.buffer.append([o, a, r, next, d])

            if start_training():
                #data = self.buffer.sample()
                # state, action, reward, next_state, done
                index = np.random.choice(len(self.buffer), cfg.batch_size)
                data = [list(i) for i in zip(*[self.buffer[i] for i in index])]
                self.learn(*data, cfg=cfg)

        def select_action(obs):
            if start_training():
                return self.actor(obs)
            else:
                return [self.action_space.sample() for i in obs]
        
        _, self.rolling_state = roller(
            envs,
            self.rolling_state,
            select_action,
            nsteps=cfg.nsteps,
            process_obs=self.process_obs,
            ignore_episode_done=cfg.ignore_episode_done,
            verbose=True,
            # pass the training func for optimization
            training_func=training_func,
        )

        if cfg.eval_episode and self.training_epoch % cfg.eval_episode == 0 and get_rank() == 0:
            assert test_envs is None, "not unit tested yet"
            self.eval_and_save(test_envs or envs)

        logger.logkvs(
            dict(
                buffer_size=len(self.buffer),
                training_epoch=self.training_epoch,
                training_steps=self.learning_steps,
                nenvs=envs.env_num,
                total_env_steps=self.num_samples,
            )
        )
        logger.dumpkvs()
