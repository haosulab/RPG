from solver.train_rpg import Trainer
from solver.envs.numerical.test_bandit_elbo import show_curve, print_elbo 


def main():
    trainer = Trainer.parse(
        env=dict(TYPE='PointEnv2', with_dt=False, save_traj=True),
        actor=dict(
            not_func=False, # Only in this case the infoq could distinguish the two?
            a_head=dict(TYPE='Normal', linear=True, squash=False, std_mode='statewise', std_scale=0.1),
        ), # initial 0.3 
        z_dim=1,
        z_cont_dim=0,
        #K=2,
        K = 0,
        batch_size=1024,
        rpg=dict(optim=dict(lr=0.0003), weight=dict(reward=1., ent_a=0.1, mutual_info=1., prior=0.1, ent_z=2.)),
        increasing_reward=0,
        #backbone=dict(dims=(256, 256)),
        info_net=dict(noise=0.2, obs_weight=0.1)
    )
    trainer.epoch_hooks.append(show_curve)
    # trainer.epoch_hooks.append(print_elbo)
    trainer.start()


if __name__ == '__main__':
    main()