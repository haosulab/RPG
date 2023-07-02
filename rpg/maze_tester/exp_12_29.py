from rpg.maze_tester.maze_exp import *

def add_var(k, d):
    assert k not in base_config['_variants']
    base_config['_variants'][k] = d

    
if __name__ == '__main__':
    add_var(
        'ant_maxentrl', dict(
            _inherit='ant_squash', 
            hidden=dict(n=1), info=dict(coef=0.0),
            head=dict(std_scale=1.0, std_mode='statewise'),
            pi_a=dict(ent=dict(coef=0.1, target_mode='none')),
            rnd=dict(scale=1.),
        )
    )
    add_var(
        'ant_maxentrl2', dict(
            _inherit='ant_maxentrl',
            pi_a=dict(ent=dict(coef=0.1, target_mode='auto', target=-2.)),
        )
    )
    add_var(
        'ant_sa', dict(
            _inherit='ant_maxentrl2',
            hidden=dict(use_next_state=False, action_weight=1.),
        )
    )

    exp = build_exp(base_config)

    # search for pi_a coef
    exp.add_exps(
        'entcoef', dict(pi_a=dict(ent=dict(coef=[1., 0.1, 0.01, 0.001]),)), 
        base='ant_maxentrl', default_env='AntMaze2',
    )

    # search for pi_a coef
    # TODO: reduce the mutual info weight?
    # exp.add_exps(
    #     'entrnd', dict(rnd=dict(scale=[1., 10., 100., 1000.],)), 
    #     base='ant_maxent', default_env='AntMaze2',
    # )


    # test various representation:
    #  - normal RL - discrete - gaussian - uniform
    exp.add_exps(
        'repr',
        dict(
            info=dict(coef=[0., 0.05, 0.01]),
            hidden=dict(TYPE=['Categorical', 'Categorical', 'Gaussian'], n=[1, 6, 5]), ), 
        base='ant_maxentrl2', default_env='AntMaze2',
    )
    exp.add_exps(
        'reprsa',
        dict(
            info=dict(coef=[0.01, 0.005]),
            hidden=dict(TYPE=['Categorical', 'Gaussian'], n=[6, 5]), ), 
        base='ant_sa', default_env='AntMaze2',
    )

    # search for suitable RND value first: reward scale, action scale + [info scale in the end]
    # exp.add_exps(
    #     'entrnd', dict(rnd=dict(scale=[1., 10., 100., 1000.],)), 
    #     base='ant_maxent', default_env='AntMaze2',
    # )

    # TODO: consider harder env.

    # TODO: test relabel

    exp.add_exps(
        'mazer', dict(
            env_cfg=dict(reward=True),
            trainer=dict(weights=dict(reward=10., q_value=10., state=1000.)),
            hidden=dict(n=[1, 1, 6, 6]),
            rnd=dict(scale=1., normalizer=['ema','none', 'ema', 'none']),
        ), base = 'small', default_env='MediumMazeR',
    )
    # TODO: test RND scale ..
    exp.add_exps(
        'mazer2', dict(
            env_cfg=dict(reward=True),
            trainer=dict(weights=dict(reward=10., q_value=10., state=1000.)),
            hidden=dict(n=[1, 1, 6, 6]),
            rnd=dict(scale=10., normalizer=['ema','none', 'ema', 'none']),
        ), base = 'small', default_env='MediumMazeR',
    )

    exp.main()