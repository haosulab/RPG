import gym
from diffrl.rl.vec_envs import SubprocVectorEnv
from diffrl.rl.utils import logger
from diffrl.rl.env_maker import make_block


logger.configure('/tmp/ppo_block')

env = SubprocVectorEnv([make_block for i in range(20)])
from diffrl.rl.agents.ppo_agent import PPOAgent

agent = PPOAgent(env.observation_space[0], env.action_space[0], nsteps=2000, eval_episode=10, **{"ppo_optim": {"max_kl": 0.1}, "evaluator_cfg": {"render_episodes": 1}}, obs_norm=False).cuda()

print('start ...')
for i in range(1000000):
    agent.train(env)