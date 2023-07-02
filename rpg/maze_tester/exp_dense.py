from rpg.maze_tester.maze_exp import *

configs = dict(
    env_name = dict(
        cabinet='Cabinet',
        ant='AntPush',
        fall='AntFall'
    )
)

if __name__ == '__main__':
    exp = build_exp(base_config)

    for env_name  in ['cabinet', 'ant', 'fall']: # ensure the experiments are finished ..

        schedule = "1000000" if env_name == 'ant' else "800000"
        exp.add_exps(
            f'dense{env_name}',
            dict(
                env_cfg=dict(n=5, reward_type='dense', obs_dim=0),
                _base=[
                    'mbsacv3', 'rpgdv3', 'rpgdv3', 'rpgdv3', 'rpgdv3', 
                    'rpgdv3', 'rpgcv2', 'rewardrpg', 'rewardrpg', 'rewardrpg',
                    'rewardrpg',
                    ],
                info=dict(coef=[0.0, 0.5, 1., 5., 10., 
                                50., 1., 1., 1., 1.,
                                1. ]),
                rnd=dict(scale=0.),
                z_delay=4,
                pi_a=dict(ent=dict(coef=1.)),
                pi_z=dict(ent=dict(coef=100.)),
                reward_schedule=["1000000" if env_name == 'ant' or env_name == 'fall' else "800000"] * 8 + [
                    '2seg(0.4,400000,600000)', '2seg(0.2,400000,600000)', '2seg(0.1,400000,600000)'],
            ),
            names=[
                'sac', 'discrete5', 'discrete10', 'discrete50', 'discrete100',
                'discrete500', 'gaussian10', 'incrR', 'seg', 'seg2', 'seg3'], # note that seg is seg2 before
            base='mbsacv3', 
            default_env = configs['env_name'][env_name],
        )
    
    exp.main()