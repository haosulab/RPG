from rpg.maze_tester.maze_exp import *

if __name__ == '__main__':
    exp = build_exp(base_config)

    exp.add_exps(
        'antcross',
        dict(
            info=dict(coef=[0., 0.05, 0.01]),
            hidden=dict(TYPE=['Categorical', 'Categorical', 'Gaussian'], n=[1, 6, 5]), ), 
        base='antcross', default_env='AntMaze3',
    )

    exp.main()