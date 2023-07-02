from rpg.maze_tester.maze_exp import *

if __name__ == '__main__':
    exp = build_exp(base_config)
    exp.path = "../../expicml/rpg_official"
    
    # exp.add_exps(
    #     f'cheetah',
    #     dict(
    #         env_cfg=dict(n=1),
    #         info = dict(coef=[0.0]),
    #         reward_scale=1.0,
    #         hooks=dict(),
    #     ),
    #     names=["test_name"],
    #     base='mbsac', 
    #     default_env='HalfCheetah-v2'
    # )
    
    exp.add_exps(
        f'rpgcv2',
        dict(
            env_cfg=dict(n=[5], reward_type=["sparse"]),
            info=dict(coef=[0.005]),
            max_total_steps=None,
            max_epoch=None,
        ),
        names=["test_name"],
        base='rpgcv2', 
        default_env="EEArm"
    )
    

    exp.add_exps(
        f"dense",
        dict(
            model=dict(gamma=[0.995]),
            env_cfg=dict(n=[5]),
            reward_schedule='2seg(0.01,400000,1000000)'
        ),
        base='rewardrpg',
        names=["test_name"],
        default_env="AntPushDense"
    )
    exp.main()

# python3 mbrpg.py --env_name EEArm --exp rpgcv2 --seed 0

# python3 mbrpg.py --env_name EEArm --exp rpgcv2 --seed 0
# --wandb False --seed 0 --id 0