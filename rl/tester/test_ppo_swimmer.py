import gym
from rl.vec_envs import SubprocVectorEnv
from tools.utils import logger


def make_env():
    return gym.make('Swimmer-v2')

logger.configure('/tmp/ppo_swimmer')

env = SubprocVectorEnv([make_env for i in range(20)])
from rl.ppo_agent import PPOAgent

# make 0.999 to 0.995 if not work ..
agent = PPOAgent(env.observation_space[0], env.action_space[0], nsteps=2000, eval_episode=100, **{"ppo_optim": {"max_kl": 0.1}, "evaluator_cfg": {"render_episodes": 1}}, gamma=0.999).cuda()

print('start ...')
for i in range(1000000):
    agent.train(env)