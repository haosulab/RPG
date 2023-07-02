from tools.config import CN, merge_a_into_b, extract_variant
from rpg.maze_tester.maze_exp import *

if __name__ == '__main__':
    exp = build_exp(base_config)

    exp.add_exps(
        f'denseantabl',
        dict(
            _variants=dict(
                _base = dict(
                    env_cfg=dict(n=5, reward_type='dense', obs_dim=0),
                    z_delay=4,
                    rnd=dict(scale=0.),

                    pi_a=dict(ent=dict(coef=1.)),
                    pi_z=dict(ent=dict(coef=100.)),
                ),
                seg3=dict(
                    _base='rewardrpg',
                    reward_schedule='2seg(0.1,400000,600000)',
                    info=dict(coef=1.),
                ),
                sac=dict(
                    _base='mbsacv3',
                    env_cfg=dict(n=5),
                    reward_schedule=None,
                ),
                largestd=dict(
                    _base='rewardrpg',
                    pi_a=dict(ent=dict(coef=1., target=-4.)),
                    reward_schedule='2seg(0.1,400000,600000)',
                    info=dict(coef=1.),
                ),
                noepsilon=dict(
                    _base='rewardrpg',
                    hidden=dict(head=dict(epsilon=0.02)),
                    reward_schedule='2seg(0.1,400000,600000)',
                ),
                seg3n1=dict(
                    _base='rewardrpg',
                    env_cfg=dict(n=1),
                    reward_schedule='2seg(0.1,400000,600000)',
                ),
                seg4=dict(
                    _base='rewardrpg',
                    reward_schedule='2seg(0.05,400000,600000)',
                ),

                seg3n1gamma=dict(
                    _base='rewardrpg',
                    env_cfg=dict(n=1),
                    reward_schedule='2seg(0.1,400000,600000)',
                    model=dict(gamma=0.995),
                    info=dict(coef=1.),
                ),
                sacn1gamma=dict(
                    _base='mbsacv3',
                    env_cfg=dict(n=1),
                    reward_schedule=None,
                    model=dict(gamma=0.995),
                ),
                seg4n1gamma=dict(
                    _base='rewardrpg',
                    env_cfg=dict(n=1),
                    reward_schedule='2seg(0.01,400000,600000)',
                    model=dict(gamma=0.995),
                ),
                seg3mix=dict(
                    _base='rpgmix',
                    use_reward_schedule=True,
                    env_cfg=dict(n=3),
                    reward_schedule='2seg(0.1,400000,600000)',
                    model=dict(gamma=0.995),
                    info=dict(coef=1.),
                ),

                seg4save=dict(
                    _base='rewardrpg',
                    env_cfg=dict(n=10),
                    reward_schedule='2seg(0.01,400000,600000)',
                    model=dict(gamma=0.995),
                    save_eval_results=True,
                    path='data/trajs/ant/seg4save',
                ),
            )
        ),
        base=None, 
        default_env ='AntPush',
    )

    exp.main()