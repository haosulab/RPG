import gym
from diffrl.rl.vec_envs import SubprocVectorEnv
from diffrl.rl.utils import logger


def make_env():
    return gym.make('HalfCheetah-v2')

logger.configure('/tmp/tmp')

env = SubprocVectorEnv([make_env for i in range(1)])
from diffrl.rl.agents.ppo_agent import PPOAgent

agent = PPOAgent(env.observation_space[0], env.action_space[0], nsteps=2000, eval_episode=100, **{"ppo_optim": {"max_kl": 0.1}, "evaluator_cfg": {"render_episodes": 1}}).cuda()

import torch
state_dict = torch.load('/tmp/diffrl-2022-01-21-08-51-00-986330/checkpoints/best.ckpt')
print(state_dict.keys())
agent.load_state_dict(state_dict)

agent.eval_and_save(env)
#print('start ...')
#for i in range(1000000):
#    agent.train(env)