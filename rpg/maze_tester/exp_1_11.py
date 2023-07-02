from rpg.maze_tester.maze_exp import *

if __name__ == '__main__':
    exp = build_exp(base_config)

    exp.add_exps(
        'hammer',
        dict(
            _base=['mbsac', 'mbsacrnd', 'rpgnormal', 'rpgdiscrete', 'mbsacrnd5', 'rpgnormal1', 'rpgdiscrete1'],
            env_cfg=dict(reward_type='sparse'),
        ),
        base=None, default_env='AdroitHammer',
        #names=['rnd', 'rl', 'rndx5', 'rnd001', 'rnd0005', 'rnd01'] + ['g0005', 'g005', 'g001', 'g01', 'g05', 'g1']
    )
    # python3 maze_tester/exp_1_11.py --exp hammer2 --runall remote --wandb True  --seed 1,2,3 --cpu 5 --silent
    # also test group
    exp.add_exps(
        'hammer2',
        dict(
            _base=['rpgnormal1', 'mbsacrnd', 'rpgsac'],
            env_cfg=dict(reward_type='sparse'),
        ),
        base=None, default_env='AdroitHammer',
        #names=['rnd', 'rl', 'rndx5', 'rnd001', 'rnd0005', 'rnd01'] + ['g0005', 'g005', 'g001', 'g01', 'g05', 'g1']
    )

    exp.add_exps(
        'cabinet',
        dict(
            _base=['rpgnormal1', 'mbsacrnd'],
            env_cfg=dict(reward_type='sparse'),
        ),
        base=None, default_env='EEArm',
        #names=['rnd', 'rl', 'rndx5', 'rnd001', 'rnd0005', 'rnd01'] + ['g0005', 'g005', 'g001', 'g01', 'g05', 'g1']
    )

    exp.add_exps(
        'stickpull',
        dict(
            _base=['mbsacrnd', 'rpgnormal'],
            env_cfg=dict(reward_type='sparse', n=5),
        ),
        base=None, default_env='MWStickPull',
    )

    exp.main()