def main():
    import os
    from solver import MODEL_PATH
#    from solver.networks import PointnetBackbone
    from solver.envs.goal_env_base import make_env
    from rl.vec_envs import SubprocVectorEnv, DummyVectorEnv
    from rl.ppo_agent import PPOAgent
    from tools.utils import logger

    import argparse
    from torch.multiprocessing import set_start_method
    set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default='GripperUmaze')
    parser.add_argument("--path", type=str, default='ppo')
    parser.add_argument("--mode", type=str, default='train')
    args, _ = parser.parse_known_args()

    if args.mode == 'train':
        logger.configure(os.path.join(MODEL_PATH, args.path), format_strs='csv+tensorboard+stdout'.split('+'))
    else:
        logger.configure(os.path.join(MODEL_PATH, args.path, args.mode))
    env = DummyVectorEnv([lambda: make_env(args.env_name) for i in range(1)])

    agent = PPOAgent.parse(
        env.observation_space[0], env.action_space[0],
        nsteps=2000, eval_episode=20, show_roller_progress=True, batch_size=50, n_epochs=5, ppo_optim=dict(max_kl=0.1),
        evaluator_cfg=dict(render_episodes=1),
        actor=dict(backbone=dict(TYPE="PointnetBackbone"), head=dict(TYPE="MaskHead")),
        obs_norm=False, reward_norm=False, parser=parser
    ).cuda()

    if args.mode == 'train':
        print('start ...')
        for i in range(1000000):
            agent.train(env)
    else:
        agent.train(env)



if __name__ == '__main__':
    main()
