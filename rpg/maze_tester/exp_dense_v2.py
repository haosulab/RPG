from rpg.maze_tester.maze_exp import *

configs = dict(
    env_name = dict(
        cabinet='Cabinet',
        ant='AntPush2',
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
                #_base='rewardrpgc',
                info=dict(coef=[0.001, 0.01, 0.1, 1., 10., 0.002, 0.003, 0.005]),
                rnd=dict(scale=0.),
                z_delay=4,
                pi_a=dict(ent=dict(coef=1.)),
                pi_z=dict(ent=dict(coef=100.)),
                reward_schedule="1000000" if env_name == 'ant' or env_name == 'fall' else "800000",
            ),
            names=['c0001', 'c001', 'c01', 'c1', 'c10', 'c002', 'c003', 'c005'], # note that seg is seg2 before
            base='rewardrpgc', 
            default_env = configs['env_name'][env_name],
        )
    
    exp.main()