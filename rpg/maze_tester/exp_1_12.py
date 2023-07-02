from rpg.maze_tester.maze_exp import *

if __name__ == '__main__':
    exp = build_exp(base_config)


    exp.add_exps(
        'block',
        dict(
            _base=['rpgcv2', 'mbsacv3'],
            env_cfg=dict(reward_type='sparse', n_block=2, n=5),
        ),
        base=None, default_env='BlockPush',
    )

    exp.add_exps(
        'stickpull',
        dict(
            _base=['rpgcv2', 'mbsacv3'],
            env_cfg=dict(reward_type='sparse', n=5),
        ),
        base=None, default_env='MWStickPull',
    )

    exp.add_exps(
        'cabinet',
        dict(
            _base=['rpgcv2', 'mbsacv3'],
            env_cfg=dict(reward_type='sparse', n=5),
        ),
        base=None, default_env='EEArm',
    )
    


    exp.add_exps(
        'peginsert',
        dict(
            _base=['rpgcv2', 'mbsacv3'],
            env_cfg=dict(reward_type='sparse', n=5),
        ),
        base=None, default_env='PegInsert',
    )

    exp.add_exps(
        'hammern5',
        dict(
            _base=['mbsacrnd', 'mbddpgrnd', 'mbsacv2rnd', 'mbsaclowstd', 'mbsacv3'],
            env_cfg=dict(reward_type='sparse', n=5),
            max_total_steps=1500000, # 1.5M
        ),
        base=None, default_env='AdroitHammer',
        #names=['rnd', 'rl', 'rndx5', 'rnd001', 'rnd0005', 'rnd01'] + ['g0005', 'g005', 'g001', 'g01', 'g05', 'g1']
    )


    exp.add_exps(
        'test_video',
        dict(
            _base=['mbsacrnd5'],
            env_cfg=dict(reward_type='sparse', n=1),
            hooks=dict(save_traj=dict(n_epoch=1)),
            max_epoch=10,
        ),
        base=None, default_env='MWStickPull',
    )
    exp.main()