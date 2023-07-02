from rpg.maze_tester.maze_exp import *

if __name__ == '__main__':
    exp = build_exp(base_config)

    # exp.add_exps(
    #     'kitcheninfo',
    #     dict(
    #         #_base=['rpgcv2', 'mbsacv3'],
    #         env_cfg=dict(reward_type='bonus', n=5),
    #         buffer=dict(max_episode_num=5000),
    #         info=dict(coef=[0.002, 0.005, 0.01, 0.0005, 0.0001, 0.0008])
    #     ),
    #     base='rpgcv2', default_env='Kitchen',
    # )

    exp.add_exps(
        'kitchenreward',
        dict(
            #_base=['rpgcv2', 'mbsacv3'],
            env_cfg=dict(reward_type='bonus', n=5),
            buffer=dict(max_episode_num=5000),
            #info=dict(coef=[0.002, 0.005, 0.01, 0.0005, 0.0001, 0.0008])
            reward_scale=[2.5, 1.],
        ),
        base='rpgcv2', default_env='Kitchen',
    )

    exp.add_exps(
        'kitchenrnd',
        dict(
            #_base=['rpgcv2', 'mbsacv3'],
            env_cfg=dict(reward_type='bonus', n=5),
            buffer=dict(max_episode_num=5000),
            #info=dict(coef=[0.002, 0.005, 0.01, 0.0005, 0.0001, 0.0008])
            #reward_scale=[2.5, 1.],
            rnd=dict(scale=[0.1, 0.2, 0.5, 1.]),
        ),
        base='rpgcv2', default_env='Kitchen',
    )

    exp.add_exps(
        'stickpull',
        dict(
            env_cfg=dict(reward_type='sparse', n=5),
            info=dict(coef=[0.002,0.005,0.01,0.0005,0.0001,0.0008])
        ),
        base='rpgcv2', default_env='MWStickPull',
    )



    exp.add_exps(
        'antpush400',
        dict(
            _base=['rpgcv2', 'mbsacv3', 'rpgdv2'],
            env_cfg=dict(reward_type='sparse', n=5),
            buffer=dict(max_episode_num=4000),
            info=dict(coef=[0.0002, 0., 0.001]),
        ),
        base=None, default_env='AntPush',
    )

    exp.add_exps(
        'kitchensac',
        dict(
            _base=['mbsacv3'],
            env_cfg=dict(n=5, reward_type='sparse'),
        ),
        base='mbsacv3', default_env='Kitchen',
    )

    for env_name  in ['cabinet', 'stickpull', 'kitchen']:
        exp.add_exps(
            f'{env_name}info',
            dict(
                env_cfg=dict(n=5, reward_type='sparse'),
                info=dict(coef=[0.002, 0.005, 0.01, 0.0008, 0.0001])
            ),
            base='rpgcv2', default_env = dict(cabinet='EEArm', stickpull='MWStickPull', kitchen='Kitchen')[env_name],
        )

    # python3 maze_tester/exp_1_14.py --exp hammerinfo --runall remote_parallel  --wandb True --cpu 5 --seed 1,2 --silent
    exp.add_exps(
        f'hammerinfo',
        dict(
            env_cfg=dict(n=5, reward_type='sparse'),
            info=dict(coef=[0., 0.001, 0.002, 0.005, 0.01]),
            max_total_steps=1500000,
        ),
        base='rpgcv2', default_env = 'AdroitHammer',
    )


    for env_name  in ['cabinet', 'stickpull', 'hammer', 'kitchen', 'ant']: # ensure the experiments are finished ..
        exp.add_exps(
            f'{env_name}baseline',
            dict(
                env_cfg=dict(n=5, reward_type='sparse'),
                _base=['rpgcv2', 'mbsacv3',
                       'rpgcv2', 'rpgdv3'],
                info = dict(coef=[0.005, 0., 0.002, 0.001])
            ),
            names=['rpg', 'sac', 'rpg002', 'rpgd'],
            base=None, default_env = dict(
                cabinet='EEArm',
                stickpull='MWStickPull',
                kitchen='Kitchen',
                hammer='AdroitHammer',
                ant='AntPush'
            )[env_name],
        )

    
    exp.add_exps(
        'smallmaze',
        dict(
            # TODO: action_scale=0.3
            reward_scale=5.,
            env_cfg=dict(n=5),
            max_epoch=1000,
            _base=['rpgdv3', 'mbsacv3', 'rewardrpg', 'rewardsac'],
            save_video=0,
            pi_z=dict(ent=dict(coef=10., target_mode='none')),
            info=dict(coef=0.1),
            reward_schedule='100000',
            z_delay=10,
        ),
        base=None, default_env='SmallMaze',
        #names=['rnd', 'vae', 'count','rndn100']
    )

    exp.main()