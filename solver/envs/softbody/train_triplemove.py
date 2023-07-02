import torch
import numpy as np
from tools.utils import logger, totensor
import solver.envs.softbody
from solver.train_rpg import Trainer
from solver.envs.numerical.test_bandit_elbo import show_curve, print_elbo 


def main():
    trainer = Trainer.parse(
        env=dict(TYPE='TripleMove', reward_weight=1.),
        actor=dict(
            not_func=True,
            a_head=dict(
                TYPE='Normal',
                linear=True,
                squash=False,
                std_mode='statewise',
                # std_mode='fix_no_grad',
                #std_scale=0.05
                std_scale = 0.2
            )
        ), # initial 0.3 
        info_net=dict(action_weight=0.),
        rpg=dict(optim=dict(lr=0.001, accumulate_grad=0), use_action_entropy=True),
        # backbone=dict(TYPE='PointNet'),
        increasing_reward = 0,

        z_dim=5, z_cont_dim=0,
        #batch_size=8,
        batch_size=32,
        n_batches=50,
        #max_epochs=30,
        max_epochs = 20,

        record_gif=True,

        path = 'tmp',
    )
    trainer.epoch_hooks.append(show_curve)
    trainer.start()


if __name__ == '__main__':
    main()