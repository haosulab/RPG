from solver.train_rpg import Trainer
from solver.envs.numerical.test_bandit_elbo import show_curve, print_elbo 


def main():
    trainer = Trainer.parse(
        env=dict(TYPE='PointEnv2', save_traj=True),
        actor=dict(
            not_func=False,
            a_head=dict(TYPE='Normal', linear=True, squash=False, std_mode='fix_no_grad', std_scale=0.1)
        ), 
        # initial 0.3 
        z_dim=0, z_cont_dim=5,
        batch_size=100,
        rpg = dict(
        weight=dict(ent_z=200.),
        ),
        max_epochs=50,
        ent_z_decay = 0.94,
    )
    trainer.epoch_hooks.append(show_curve)
    trainer.epoch_hooks.append(print_elbo)
    trainer.start()


if __name__ == '__main__':
    main()