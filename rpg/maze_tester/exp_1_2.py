from rpg.maze_tester.maze_exp import *

if __name__ == '__main__':
    exp = build_exp(base_config)

    exp.add_exps(
        'antcross2',
        dict(
            info=dict(coef=[0., 0.1, 0.05, 0.01, 0.01, 0.005]),
            hidden=dict(TYPE=['Categorical', 'Categorical', 'Categorical', 'Categorical', 'Gaussian', 'Gaussian'], n=[1, 6, 6, 6, 5, 5]), ), 
        base='antcross', default_env='AntMaze3',
    )

    exp.main()