import multiprocessing as mp



if __name__ == '__main__':
    from diffrl.rl.env_maker import make_move
    mp.set_start_method('spawn')
    from diffrl.rl.vec_envs import SubprocVectorEnv
    env = SubprocVectorEnv([make_move for i in range(1)])

    env.reset()
    for i in range(100):
        _, r, d, _ = env.step([i.sample() for i in env.action_space])
        print(i, r, d)
        if d[0]:
            env.reset()

    from diffrl.rl.utils import logger
    logger.configure()

    from diffrl.rl.agents.ppo_agent import PPOAgent
    agent = PPOAgent(env.observation_space[0], env.action_space[0], nsteps=100, compute_gradient=True, n_epochs=0).cuda()

    print('start ...')
    for i in range(1000000):
        agent.train(env, show_roller_progress=True)