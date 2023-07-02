import gym
import os

from rl.vec_envs import SubprocVectorEnv
from tools.utils import logger


def make_env():
    return gym.make('HalfCheetah-v3')

logger.configure('/tmp/ppo_cheetah')

env = SubprocVectorEnv([make_env for i in range(20)])
from rl.ppo_agent import PPOAgent

agent = PPOAgent(env.observation_space[0], env.action_space[0], nsteps=2000, eval_episode=50, **{"ppo_optim": {"max_kl": 0.1}, "evaluator_cfg": {"render_episodes": 1}}, actor=dict(head=dict(std_mode='fix_no_grad'))).cuda()

print('start ...')
for i in range(1000000):
    agent.train(env)