
from diffrl.rl.agents.ppo_agent import PPOAgent
import gym
from diffrl.rl.vec_envs import SubprocVectorEnv
from diffrl.rl.utils import logger


def make_env():
    return gym.make('HalfCheetah-v2')


logger.configure('/tmp/ppg_cheetah')

env = SubprocVectorEnv([make_env for i in range(20)])

agent = PPOAgent(
    env.observation_space[0],
    env.action_space[0],
    nsteps=2000,
    eval_episode=50,
    **{
        "ppo_optim": {
            "max_kl": 0.1
        },
        "evaluator_cfg": {
            "render_episodes": 1
        },
        "soft_ppo": True,
        "actor.head.std_mode": "statewise",
    }
).cuda()

print('start ...')
for i in range(1000000):
    agent.train(env)
