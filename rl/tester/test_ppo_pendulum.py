
import gym
from diffrl.rl.vec_envs import SubprocVectorEnv
from diffrl.rl.utils import logger

def make_env():
    return gym.make('InvertedPendulum-v2')


logger.configure()

env = SubprocVectorEnv([make_env for i in range(5)])


from diffrl.rl.agents.ppo_agent import PPOAgent

agent = PPOAgent(env.observation_space[0], env.action_space[0], nsteps=2000).cuda()

print('start ...')
for i in range(1000000):
    agent.train(env)