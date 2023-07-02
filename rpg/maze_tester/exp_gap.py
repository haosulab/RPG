from tools.config import CN, merge_a_into_b, extract_variant
from rpg.maze_tester.maze_exp import *

if __name__ == '__main__':
    exp = build_exp(base_config)

    exp.add_exps(
        f'gapexplore',
        dict(
            _variants=dict(
                _base = dict(
                    reward_scale=0., #float((name=='reward'))*5.,
                    env_cfg=dict(n=5),
                    max_epoch=1000,
                    save_video=0,
                    pi_z=dict(ent=dict(coef=10., target_mode='none')),
                    reward_schedule='40000',
                    z_delay=5,
                ),
                discrete=dict(
                    _base='rpgdv3',
                    info=dict(coef=0.1),
                ),
                gaussian=dict(
                    _base='rpgcv2',
                    info=dict(coef=0.01),
                ),
                mbsac=dict(
                    _base='mbsacv3',
                ),

                mbsaclargestd0=dict(
                    _base='mbsacv3',
                    pi_a=dict(ent=dict(target=0.)),
                ),
                mbsaclargestd1=dict(
                    _base='mbsacv3',
                    pi_a=dict(ent=dict(target=-1.)),
                ),
                discrete05=dict(
                    _base='rpgdv3',
                    info=dict(coef=0.05),
                ),
                discrete01=dict(
                    _base='rpgdv3',
                    info=dict(coef=0.01),
                ),

                # ablation on info
                gaussian05=dict(
                    _base='rpgcv2',
                    info=dict(coef=0.05),
                ),
                gaussian10=dict(
                    _base='rpgcv2',
                    info=dict(coef=0.1),
                ),

                gaussian0=dict(
                    _base='rpgcv2',
                    info=dict(coef=0.0),
                ),
                gaussian005=dict(
                    _base='rpgcv2',
                    info=dict(coef=0.005),
                ),
                gaussian0001=dict(
                    _base='rpgcv2',
                    info=dict(coef=0.0001),
                ),

                # ablation on hidden
                gaussiand1=dict(
                    _base='rpgcv2',
                    info=dict(coef=0.01),
                    hidden=dict(n=1),
                ),
                gaussiand3=dict(
                    _base='rpgcv2',
                    info=dict(coef=0.01),
                    hidden=dict(n=3),
                ),
                gaussiand6=dict(
                    _base='rpgcv2',
                    info=dict(coef=0.01),
                    hidden=dict(n=6),
                ),

                mpc=dict(_base='mpc'),
                sacgmm=dict(_base='sacgmm'),
                sacflow=dict(_base='sacflow'),
                # compare with other heads

                rnddim0=dict(
                    _base='rpgcv2',
                    info=dict(coef=0.01),
                    env_cfg=dict(obs_dim=0)
                ),
                rnddim2=dict(
                    _base='rpgcv2',
                    info=dict(coef=0.01),
                    env_cfg=dict(obs_dim=2),
                ),
                rndbuf=dict(
                    _base='rpgcv2',
                    info=dict(coef=0.01),
                    buffer=dict(max_capacity=2000,),
                ),
                rndnonorm=dict(
                    _base='rpgcv2',
                    info=dict(coef=0.01),
                    rnd=dict(density=dict(normalizer='none')),
                ),
                #rnd=dict(),

                gaussian_save=dict(
                    _base='rpgcv2',
                    info=dict(coef=0.01),
                    path="/cephfs/hza/buffers/gaussian",
                    save_model_epoch=100,
                ),
                mbsac_save=dict(
                    _base='mbsacv3',
                    path="/cephfs/hza/buffers/mbsac",
                    save_buffer_epoch=100,
                ),

                mbsac_save_local=dict(
                    _base='mbsacv3',
                    path="data/buffers/mbsac",
                    save_model_epoch=50,
                ),
                gaussian_save_local=dict(
                    _base='rpgcv2',
                    info=dict(coef=0.01),
                    path="data/buffers/gaussian",
                    save_model_epoch=50,
                    
                ),

                mbsacn1=dict(
                    _base='mbsacv3',
                    env_cfg=dict(n=1),
                ),
                gaussiann1=dict(
                    _base='rpgcv2',
                    info=dict(coef=0.01),
                    env_cfg=dict(n=1),
                ),
            )
        ), 
        base=None, default_env='GapMaze',
    )

    exp.main()