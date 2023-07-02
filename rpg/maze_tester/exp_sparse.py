from rpg.maze_tester.maze_exp import *

configs = dict(
    env_name = dict(
        cabinet='EEArm',
        stickpull='MWStickPull',

        kitchen='Kitchen',
        kitchen2='Kitchen2',
        kitchen3='Kitchen3',
        kitchen4='Kitchen4',

        hammer='AdroitHammer',
        ant='AntPush',
        ant2='AntPush2',
        block='BlockPush2',
        fall='AntFall',
        block3='BlockPush',
        door='AdroitDoor',
        ball='MWBasketBall',
    )
)

if __name__ == '__main__':
    exp = build_exp(base_config)

    for env_name  in ['cabinet', 'stickpull', 'hammer', 'kitchen', 'ant', 'block', 'fall', 'block3', 'door', 'ball', 'ant2', 'kitchen2', 'kitchen3', 'kitchen4']: # ensure the experiments are finished ..
        exp.add_exps(
            f'{env_name}',
            dict(
                env_cfg=dict(n=5, reward_type='sparse'),
                _base=['mbsacv3', 'rpgcv2', 'rpgcv2', 'rpgcv2', 'rpgcv2', 'rpgdv3', 'rpgdv3', 'rpgdv3', 'rpgdv3', 'rpgdv3', 'rpgcv2', 'rpgdelay',
                       'rpgdelay',
                       ],
                info = dict(coef=[0.0, 0.002, 0.005, 0.001, 0.0005,
                                  0.001, 0.005, 0.0005, 0.002, 0.0001,
                                  0.0001, 0.005, 0.0005])
            ),
            names=['sac', 'gaussian002', 'gaussian005', 'gaussian001', 'gaussian0005', 'discrete001', 'discrete005', 'discrete0005', 'discrete002', 'discrete0001', 'gaussian0001', 'delay005', 'delay0005'],
            base='mbsacv3', 
            default_env = configs['env_name'][env_name],
        )
    
    exp.main()