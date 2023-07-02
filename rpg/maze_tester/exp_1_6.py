from rpg.maze_tester.maze_exp import *

if __name__ == '__main__':
    exp = build_exp(base_config)

    exp.add_exps(
        'antfork',
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
                TYPE='Categorical',
                n=[12, 1, 12]
            ),
            info=dict(coef=[0.03, 0.0, 0.03]),
            env_cfg=dict(n=[1, 1, 5]),
        ),
        base='antcross', default_env='AntFork',
        names=['rnd', 'rl', 'rndx5']
    )

    exp.add_exps(
        'blocknormal',
        dict(
            save_video=300,
            steps_per_epoch=120, 
            hooks=dict(save_traj=dict(n_epoch=20)),
            env_cfg=dict(
                n_block=3,
                n=[1] * 5 + [5, 5],
            ),
            hidden=dict(
                n=[12] * 6 + [1],
                TYPE=['Categorical', 'Gaussian', 'Gaussian', 'Gaussian', 'Gaussian', 'Gaussian', 'Categorical'],
            ),
            info=dict(coef=[0.01, 0.005, 0.001, 0.01, 0.05, 0.001, 0.0]),
        ),
        base='block', default_env='BlockPush',
    )

    exp.main()