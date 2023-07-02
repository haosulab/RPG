from rpg.maze_tester.maze_exp import *


if __name__ == '__main__':
    exp = build_exp(base_config)

    exp.add_exps(
        f'cheetah',
        dict(
            env_cfg=dict(n=5, reward_type='sparse'),
            info = dict(coef=[0.001])
        ),
        names=["test_name"],
        base='rpgcv2', 
        default_env='HalfCheetah-v2',
    )

    exp.main()


# python test_exp.py --exp cabinet --seed 0 --runall local