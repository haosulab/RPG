import os
os.environ["OMP_NUM_THREADS"] = "1" # it can accelerate the training?
from diffrl.rl.agents.ppo_agent import PPOAgent
from diffrl import skillnet
from diffrl.rl.vec_envs import SubprocVectorEnv
from diffrl.rl.env_maker import make_carbinet
from diffrl.rl.utils import logger


logger.configure('/tmp/ppg_carbinet')

env = SubprocVectorEnv([make_carbinet for i in range(10)])

agent = PPOAgent(
    env.observation_space[0],
    env.action_space[0],
    nsteps=2000,
    eval_episode=5,
    show_roller_progress=True,
    batch_size=400,
    n_epochs=5,
    **{
        "ppo_optim": {
            "max_kl": 0.1
        },
        "evaluator_cfg": {
            "render_episodes": 1
        },
        "actor.backbone.TYPE": "PointNetV0",
        "obs_norm": True,
        "actor.head.std_mode": "fix_no_grad",

        "aux_optim": {
            'mode': 'value',
            'ppo_epoch': 4,
        }
    }).cuda()

print('start ...')
for i in range(1000000):
    agent.train(env)
