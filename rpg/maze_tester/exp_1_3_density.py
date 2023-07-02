from rpg.maze_tester.maze_exp import *

if __name__ == '__main__':
    exp = build_exp(base_config)
    

    # exp.add_exps(
    #     'antcross2',
    #     dict(
    #         info=dict(coef=[0., 0.1, 0.05, 0.01, 0.01, 0.005]),
    #         hidden=dict(TYPE=['Categorical', 'Categorical', 'Categorical', 'Categorical', 'Gaussian', 'Gaussian'], n=[1, 6, 6, 6, 5, 5]), ), 
    #     base='antcross', default_env='AntMaze3',
    # )

    exp.add_exps(
        'smallmaze',
        dict(
            reward_scale=0.,
            env_cfg=dict(n=[1,1,1,100]),
            max_epoch=100,
            rnd=dict(
                density=dict(
                    TYPE=['RND', 'VAE', 'Count','RND'],
                    normalizer='ema',
                ),
                scale = [0.1, 0.003, 0.01, 0.1],
            ),
        ),
        base='small', default_env='SmallMaze',
        names=['rnd', 'vae', 'count','rndn100']
    )

    
    # python3 maze_tester/exp_1_3_density.py --exp antcross3 --seed 1,2,3 --runall remote --wandb True --silent
    exp.add_exps(
        'antcross3',
        dict(
            reward_scale=0.,
            rnd=dict(
                density=dict(
                    TYPE=['RND', 'RND', 'RND', 'VAE', 'VAE'],
                    normalizer='ema',
                ),
                scale = [0.1, 0.1, 0.1, 0.1, 0.1],
            ),
            hidden=dict(
                TYPE=['Categorical', 'Categorical', 'Gaussian', 'Categorical',  'Gaussian'],
                n=[1, 6, 5, 6, 5]
            ),
            info=dict(coef=[0., 0.05, 0.01, 0.05, 0.01]),
        ),
        base='antcross', default_env='AntMaze3',
        names=['rndrl', 'rnddiscrete', 'rndnormal','vaediscrete', 'vaenormal']
    )

    # configs = dict(
    #     rnd=dict(update_step=5),
    # )
    # exp.add_exps(
    # )

    exp.main()