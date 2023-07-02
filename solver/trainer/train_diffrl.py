def main():
    from solver.diff_agent import RLAlgo

    engine = RLAlgo.parse(
        env=dict(TYPE='UMaze'),
        agent=dict(
            num_trajs_per_epoch=5,
            policy_optim=dict(
                batch_size=5,
                training_iter=5,
                compute_value=0.,  # do not use the value here
                lr=0.0005,
            ),
            value_optim=dict(training_iter=0),
            head_cfg=dict(std_scale=0.1),
        ),
        path='rl_tmp'
    )

    engine.main()


if __name__ == '__main__':
    main()
