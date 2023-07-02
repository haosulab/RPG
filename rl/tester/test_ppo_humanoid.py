import gym
from diffrl.rl.vec_envs import SubprocVectorEnv
from diffrl.rl.utils import logger


def make_env():
    return gym.make('Humanoid-v3')

logger.configure('/tmp/ppo_humanoid')

env = SubprocVectorEnv([make_env for i in range(20)])
from diffrl.rl.agents.ppo_agent import PPOAgent

agent = PPOAgent(env.observation_space[0], env.action_space[0], nsteps=2000, eval_episode=100, **{"ppo_optim": {"max_kl": 0.1}, "evaluator_cfg": {"render_episodes": 1}}).cuda()

print('start ...')
for i in range(1000000):
    agent.train(env)