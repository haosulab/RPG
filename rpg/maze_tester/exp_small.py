# python3 maze_tester/test_maze.py --var relabel --env_name SmallMaze2 --reward_scale 0. --save_video 0 --path xxx --info.coef 0.01 --env_cfg.n 5 --hooks.save_traj.n_epoch 10 --max_epoch 2000 --relabel_latent None --info_delay 1 --hidden.head.std_mode fix_no_grad
from rpg.maze_tester.maze_exp import *

if __name__ == '__main__':
    exp = build_exp(base_config)

    
    for name in ['reward', 'explore']:
        exp.add_exps(
            f'small{name}',
            dict(
                # TODO: action_scale=0.3
                reward_scale=float((name=='reward'))*5.,
                env_cfg=dict(n=5),
                max_epoch=2000,
                _base=['rpgdv3', 'mbsacv3', 'rewardrpg', 'rewardsac', 'mpc', 'sacgmm', 'sacflow', 'rpgcv2', 'rpgdelay', 'rpgdelay'],
                save_video=0,
                pi_z=dict(ent=dict(coef=10., target_mode='none')),
                info=dict(coef=[0.1] * 7 + [0.02] * 2 + [0.1]),
                reward_schedule='40000',
                z_delay=[10] * 7 + [1] * 3,
            ),
            base=None, default_env='SmallMaze2',
        )



    exp.main()