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
            f'draw{env_name}',
            dict(
                env_cfg=dict(n=10, reward_type='sparse'),
                _base='rpgcv2',
                info = dict(coef=0.005),
                save_eval_results=True,
                path=[None, f"/cephfs/hza/buffers/draw{env_name}/"]
            ),
            names=['drawlocal', 'drawremote'],
            base='mbsacv3', 
            default_env = configs['env_name'][env_name],
        )
    
    exp.main()