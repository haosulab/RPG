import gym
from tools.utils import logger
from rl.vec_envs import SubprocVectorEnv
from rl.sac_agent import SACAgent
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--nenvs", type=int, default=1)
parser.add_argument("--save_path", type=str, default='/tmp/sac_cheetah')
args, _ = parser.parse_known_args()

def make_env():
    return gym.make('HalfCheetah-v3')

logger.configure(args.save_path)

env = SubprocVectorEnv([make_env for i in range(args.nenvs)])

agent = SACAgent.parse(
    env.observation_space[0],
    env.action_space[0],
    nsteps=None,
    eval_episode=50,
    **dict(
        evaluator_cfg=dict(
            render_episodes=1
        )
    ),
    parser=parser,
).cuda()

print('start ...')
for i in range(1000000):
    agent.train(env)
