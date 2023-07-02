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
        'antpush300',
        dict(
            _base=['rpgcv2', 'mbsacv3', 'rpgdv2'],
            env_cfg=dict(reward_type='sparse', n=5),
            buffer=dict(max_episode_num=4000),
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

    exp.main()