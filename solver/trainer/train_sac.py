def main():
    import gym
    import os
    import argparse

    from solver.networks import PointnetBackbone
    from solver.goal_env import make_env
    from rl.vec_envs import SubprocVectorEnv, DummyVectorEnv
    from rl.sac_agent import SACAgent
    from tools.utils import logger
    from torch.multiprocessing import set_start_method

    from solver import MODEL_PATH


    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default='GripperUmaze')
    parser.add_argument("--path", type=str, default='sac')
    args, _ = parser.parse_known_args()

    logger.configure(os.path.join(MODEL_PATH, args.path), format_strs='csv+tensorboard+stdout'.split('+'))
    env = DummyVectorEnv([lambda: make_env(args.env_name) for i in range(1)])

    agent = SACAgent.parse(
        env.observation_space[0],
        env.action_space[0],
        nsteps=None,
        eval_episode=50,
        actor=dict(backbone=dict(TYPE="PointnetBackbone"), head=dict(TYPE="MaskHead", std_mode='statewise')),
        **dict(
            evaluator_cfg=dict(
                render_episodes=1
            )
        ),
        start_step=128,
        parser=parser
    ).cuda()

    print('start ...')
    for i in range(5000000):
        agent.train(env)

if __name__ == '__main__':
    main()