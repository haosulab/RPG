from diffrl.rl.agents.ppo_agent import PPOAgent
from diffrl.rl.vec_envs import SubprocVectorEnv
from diffrl.rl.utils import logger
from diffrl.rl.env_maker import make_block
"""[summary]
Relative observe:
- simple method would work
Not relative:
- it requires fix exploration
Partial
- even huge exploration, it won't work..
    - without ppg ..
"""


logger.configure('/tmp/ppo_block_fix')

env = SubprocVectorEnv(
    [
        lambda: make_block(
            'partial',
            with_timestep=True
        ) for i in range(20)
    ]
)

agent = PPOAgent(
    env.observation_space[0],
    env.action_space[0],
    nsteps=2000,
    eval_episode=10,
    ppo_optim={"max_kl": 0.1},
    evaluator_cfg={"render_episodes": 1},
    obs_norm=False,
    #soft_ppo=True,
    actor=dict(head=dict(std_mode='fix_no_grad')),
    aux_optim={
        'mode': 'value',
        'ppo_epoch': 4,
    }
).cuda()

print('start ...')
for i in range(1000000):
    agent.train(env)
