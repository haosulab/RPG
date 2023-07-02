# python3 train_rnd.py --env.TYPE HardPush --agent.policy_optim.lr 0.01 --agent.num_trajs_per_epoch 1 --agent.render_epoch 50
def main():
    from solver.diff_agent import RLAlgo

    engine = RLAlgo.parse(
        env=dict(TYPE='ToyEnv'),
        #env=dict(TYPE='ToyEnv', task_reward=100.),
        #env=dict(TYPE='HardPush', sim_cfg=dict(max_steps=1024)),
        agent=dict(
            num_trajs_per_epoch=10,
            #num_trajs_per_epoch=1,
            policy_optim=dict(
                batch_size=1,
                compute_value=0.,  # do not use the value here
                lr=0.01, #0.01 for HardPush
                policy_gradient=1.,
                gd = 1.,
                action_penalty=0.5,
            ),
            value_optim=dict(training_iter=0),
            #rnd_optim=dict(maxlen=3000, training_iter=1),
            rnd_optim=dict(maxlen=300000, training_iter=1), #, use_embed=8),

            backbone=dict(TYPE='MLP2'),
            #head_cfg=dict(std_scale=0.6, std_mode='fix_no_grad'),
            head_cfg=dict(std_scale=0.1, std_mode='fix_no_grad', use_gmm=3),
            #head_cfg=dict(std_scale=0.2, TYPE='FixPolicy', std_mode='fix'),
            use_rnd=1.,
            render_epoch=10,

            entropy_alpha=0.00001,
        ),
        path='rl_tmp',
        trajopt=True
    )

    engine.main(300)


if __name__ == '__main__':
    main()
