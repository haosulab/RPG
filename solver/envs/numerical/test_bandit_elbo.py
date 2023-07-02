from solver.train_rpg import Trainer


def show_curve(trainer: Trainer, locals):
    actor = trainer.rpg.actor
    if actor._cfg.not_func:
        actor.render(print)


def print_elbo(trainer: Trainer, locals, log_p_o=[None], batch_size=10000, epoch=10):
    if locals['epoch_id'] % epoch == 0:
        rpg = trainer.rpg
        env = trainer.env
        log_p_o[0]  = rpg.evaluate_elbo(env, batch_size=1000, n_batch=100, log_p_o=log_p_o[0])


def main():
    trainer = Trainer.parse(
        env=dict(TYPE='Bandit'),
        actor=dict(
            not_func=True,
            a_head=dict(TYPE='Normal', linear=True, squash=False, std_mode='fix_no_grad', std_scale=0.15)
        ), # initial 0.3 
        z_dim=2, z_cont_dim=0,
        batch_size=1000,
    )
    trainer.epoch_hooks.append(show_curve)
    trainer.epoch_hooks.append(print_elbo)
    trainer.start()


if __name__ == '__main__':
    main()