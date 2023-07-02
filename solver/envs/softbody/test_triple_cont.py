def prog():
    import torch
    import numpy as np
    from tools.utils import logger, totensor
    import solver.envs.softbody
    from solver.envs.softbody import TripleMove
    from solver.train_rpg import Trainer
    from solver.envs.numerical.test_bandit_elbo import show_curve, print_elbo 

    class TripleCont(TripleMove):
        def __init__(self, cfg=None):
            super().__init__()

        def step(self, action):
            from tools.utils import clamp
            return super().step(clamp(action, -1, 1))


    def main(**kwargs):
        T = 0.01
        trainer = Trainer.parse(
            env=dict(TYPE='TripleCont', reward_weight=1.),
            actor=dict(
                not_func=False,
                a_head=dict(
                    TYPE='Normal',
                    linear=True,
                    squash=False,
                    #std_mode='statewise',
                    std_mode='fix_no_grad',
                    #std_scale=0.05
                    std_scale = 0.01
                )
            ), # initial 0.3 
            info_net=dict(action_weight=0.),
            rpg=dict(
                optim=dict(lr=0.0003),
                # use_action_entropy=True,
                weight=dict(reward=1000.0 * T, ent_a=1. *T, ent_z=1 * T, mutual_info=1. * T) # dwongrade the gradient
            ),
            # backbone=dict(TYPE='PointNet'),
            increasing_reward = 1,

            z_dim=0, z_cont_dim=10,
            #batch_size=8,
            batch_size=500,
            n_batches=50,
            #max_epochs=30,
            max_epochs = 200,

            _update = kwargs,

            record_gif=True,

            path = 'tmp',
        )
        trainer.epoch_hooks.append(show_curve)
        trainer.start()


    #main(rpg=dict(optim=dict(accumulate_grad=5)), batch_size=25, n_batches=250)
    main(max_epochs=80, rpg=dict(weight=dict(reward=800. * 0.01)))

if __name__ == '__main__':
    #main()
    #main(rpg=dict(weight=dict(prior=0.0001)))
    #prog()
    from tools.dist_utils import launch
    # launch(prog, defualt_MPI_SIZE=4, default_device='0,1')
    prog()