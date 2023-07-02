from rpg.maze_tester.maze_exp import *

if __name__ == '__main__':
    exp = build_exp(base_config)

    exp.add_exps(
        'antfork2',
        dict(
            reward_scale=0.,
            rnd=dict(
                density=dict(
                    TYPE='RND',
                    normalizer='ema',
                ),
                scale = 0.1,
            ),
            hidden=dict(
                TYPE=['Categorical'] * 6 + ['Gaussian'] * 6,
                n=[12, 1, 12, 12, 12, 12] + [12] * 6
            ),
            info=dict(coef=[0.03, 0.0, 0.03, 0.01, 0.005, 0.1, 0.0005, 0.005, 0.001, 0.01, 0.05, 0.1]),
            env_cfg=dict(n=[1, 1, 5, 1, 1, 1] + [1] * 6),
            #save_video=100,
        ),
        base='antcross', default_env='AntFork',
        names=['rnd', 'rl', 'rndx5', 'rnd001', 'rnd0005', 'rnd01'] + ['g0005', 'g005', 'g001', 'g01', 'g05', 'g1']
    )


    exp.main()