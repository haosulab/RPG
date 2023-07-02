import torch
import numpy as np
from tools.utils import logger, totensor
import solver.envs.softbody
from solver.train_rpg import Trainer, InfoNet
from solver.envs.numerical.test_bandit_elbo import show_curve, print_elbo 


def render(trainer, local_vars):
    # render trajectory for different z 
    # print(local_vars.keys())
    # env = local_vars['env']
    epoch_id = local_vars['epoch_id']
    env = trainer.env
    actor = trainer.rpg.actor

    # from IPython import embed; embed()
    images = []

    with torch.no_grad():
        import tqdm
        for i in tqdm.trange(actor._pi_z.action_space.n):
            #print(i)
            z = totensor([i], device='cuda:0', dtype=torch.long)
            obs = env.reset()
            prior_z = trainer.rpg.prior_actor.pi_z(obs, None, timestep=0).sample()[0] 

            p_z = actor.pi_z(obs, prior_z, timestep=0)
            print(z, p_z.log_prob(z))

            tmp = []
            for t in range(env.low_steps):
                p_a = actor.pi_a(obs, z, timestep=t)

                a, log_p_a = p_a.rsample()
                print(a, z)
                obs, r, _, _ = env.step(a)
                tmp.append(env.render(mode='rgb_array', render_plt=True, spp=1))
            
            images.append(tmp)

        im = []
        for t in range(env.low_steps):
            # s = None
            # for i in images:
            #     if s is None:
            #         s = i[t]
            #     else: 
            #         s = np.maximum(s, i[t])

            s = 0
            for i in images:
                s = s + i[t].astype(np.float32)
            s = s / len(images)

            s = s.clip(0, 255).astype(np.uint8)
            im.append(s)
            # tmp = []
            # for i in images:
            #     tmp.append(i[t])
            # im.append(np.concatenate(tmp, axis=1))


        logger.animate(im, f'traj{epoch_id}.mp4')


class InfoNet2(InfoNet):
    def __init__(self, obs_space, action_space, z_space, cfg=None):
        super().__init__(obs_space, action_space, z_space, cfg=cfg)

    def forward(self, s, a, z, *, timestep=None):
        # print(obs.shape, action.shape, z.shape)
        # from IPython import embed; embed()
        from tools.utils import dmul
        dist = self.main(
            (dmul(s, self._cfg.obs_weight),
             (a + torch.randn_like(a) * self._cfg.noise) * self._cfg.action_weight)
        )
        # print(dist.dist.probs, z)
        return dist.log_prob(z)


def main(**kwargs):
    from tools.config import CN
    trainer = Trainer.parse(
        env=dict(TYPE='PlbEnv'),
        actor=dict(
            not_func=False,
            a_head=dict(
                TYPE='Normal',
                linear=True, # believe in tanh
                squash=False,
                std_mode='fix_no_grad',
                std_scale=0.15
            ),
            ignore_previous_z=True,
        ), # initial 0.3 

        info_net=dict(TYPE='InfoNet2'),

        rpg=dict(optim=dict(accumulate_grad=0)),
        backbone=dict(TYPE='PointNet'),
        z_dim=4, z_cont_dim=0,
        batch_size=1,
        n_batches=50,

        record_gif=False,
        # cfg=CN(kwargs),
        _update = kwargs,
        path = 'exp/plb/rpg',
    )
    trainer.epoch_hooks.append(show_curve)
    # trainer.epoch_hooks.append(print_elbo) # do not print elbo ..
    trainer.epoch_hooks.append(render)
    trainer.start()

    del trainer


if __name__ == '__main__':
    #main()
    from tools.config import CN
    base = dict(z_dim=1, rpg=dict(optim=dict(accumulate_grad=0)))
    #main(z_dim=1, )
    #main(z_dim=1, env=dict(low_steps=20), rpg=dict(weight=dict(ent_a=0., ent_z=0., mutual_info=0., prior=0.), optim=dict(accumulate_grad=0)))
    #main(z_dim=1, env=dict(low_steps=20), rpg=dict(weight=dict(reward=20.), optim=dict(accumulate_grad=0)))
    #main(z_dim=3, env=dict(low_steps=20), rpg=dict(weight=dict(reward=20.), optim=dict(accumulate_grad=0)))
    #main(z_dim=10, env=dict(low_steps=1), rpg=dict(weight=dict(reward=0., ent_z=30., ent_a=0., mutual_info=0., prior=0.), optim=dict(accumulate_grad=0), use_action_entropy=False), n_batches=200)
    #main(z_dim=10, env=dict(low_steps=20), rpg=dict(weight=dict(reward=1., ent_z=1.), optim=dict(accumulate_grad=0), use_action_entropy=True), n_batches=200)

    def test_effects():
        main(z_dim=10, env=dict(low_steps=20), rpg=dict(weight=dict(reward=1., ent_z=1., ent_a=0.0, mutual_info=0.), optim=dict(accumulate_grad=0), use_action_entropy=True), n_batches=200, path='exp/plb/0', max_epochs=2)
        main(z_dim=10, env=dict(low_steps=20), rpg=dict(weight=dict(reward=1., ent_z=1., ent_a=0.0, mutual_info=1.), optim=dict(accumulate_grad=0), use_action_entropy=True), n_batches=200, path='exp/plb/1', max_epochs=2)
        main(z_dim=10, env=dict(low_steps=20), rpg=dict(weight=dict(reward=1., ent_z=1., ent_a=1.0, mutual_info=0.), optim=dict(accumulate_grad=0), use_action_entropy=True), n_batches=200, path='exp/plb/2', max_epochs=2)

    #main(z_dim=10, env=dict(low_steps=20), rpg=dict(weight=dict(reward=1., ent_z=1., ent_a=1.0, mutual_info=0., prior=0.01), optim=dict(accumulate_grad=0), use_action_entropy=True), n_batches=200, path='exp/plb/3', max_epochs=2)
    #main(z_dim=10, env=dict(low_steps=20), rpg=dict(weight=dict(reward=10., ent_z=1., ent_a=1.0, mutual_info=10., prior=0.03), optim=dict(accumulate_grad=0), use_action_entropy=True), n_batches=200, path='exp/plb/3', max_epochs=100)
    def test_reward():
        main(z_dim=10, env=dict(low_steps=20, dist_weight=1.), rpg=dict(weight=dict(reward=1.0, ent_z=1., ent_a=1.0, mutual_info=1., prior=0.01), optim=dict(accumulate_grad=0), use_action_entropy=False), n_batches=200, path='exp/plb/3', max_epochs=10)
        main(z_dim=10, env=dict(low_steps=20, dist_weight=2.), rpg=dict(weight=dict(reward=1.0, ent_z=1., ent_a=1.0, mutual_info=1., prior=0.01), optim=dict(accumulate_grad=0), use_action_entropy=False), n_batches=200, path='exp/plb/4', max_epochs=10)
        main(z_dim=10, env=dict(low_steps=20, dist_weight=5.), rpg=dict(weight=dict(reward=1.0, ent_z=1., ent_a=1.0, mutual_info=1., prior=0.01), optim=dict(accumulate_grad=0), use_action_entropy=False), n_batches=200, path='exp/plb/5', max_epochs=10)
        main(z_dim=10, env=dict(low_steps=20, dist_weight=10.), rpg=dict(weight=dict(reward=1.0, ent_z=1., ent_a=1.0, mutual_info=1., prior=0.01), optim=dict(accumulate_grad=0), use_action_entropy=False), n_batches=200, path='exp/plb/6', max_epochs=10)
        #main(z_dim=10, env=dict(low_steps=20), rpg=dict(weight=dict(reward=0.01, ent_z=1., ent_a=1.0, mutual_info=10., prior=0.03), optim=dict(accumulate_grad=0), use_action_entropy=True), n_batches=200, path='exp/plb/4', max_epochs=2)
        #main(z_dim=10, env=dict(low_steps=20), rpg=dict(weight=dict(reward=0.05, ent_z=1., ent_a=1.0, mutual_info=10., prior=0.03), optim=dict(accumulate_grad=0), use_action_entropy=True), n_batches=200, path='exp/plb/5', max_epochs=2)

    #test_reward()

    prior = 0.005

    #main(z_dim=10, env=dict(low_steps=40, dist_weight=5.), rpg=dict(weight=dict(reward=1.0, ent_z=1., ent_a=1.0, mutual_info=1., prior=prior), optim=dict(accumulate_grad=0), use_action_entropy=False), n_batches=200, path='exp/plb/0', max_epochs=40)
    #main(z_dim=10, env=dict(low_steps=40, dist_weight=5.), rpg=dict(weight=dict(reward=1.0, ent_z=1., ent_a=1.0, mutual_info=1., prior=prior), optim=dict(accumulate_grad=10), use_action_entropy=False), n_batches=2000, path='exp/plb/1', max_epochs=40)
    #main(z_dim=1, env=dict(low_steps=40, dist_weight=5.), rpg=dict(weight=dict(reward=1.0, ent_z=1., ent_a=1.0, mutual_info=1., prior=0.005), optim=dict(accumulate_grad=0), use_action_entropy=False), n_batches=200, path='exp/plb/2', max_epochs=40)
    #main(z_dim=10, env=dict(low_steps=40, dist_weight=5.), rpg=dict(weight=dict(reward=1.0, ent_z=1., ent_a=1.0, mutual_info=1., prior=prior), optim=dict(accumulate_grad=10), use_action_entropy=False), n_batches=2000, path='exp/plb/3', max_epochs=40)
    #main(z_dim=10, env=dict(low_steps=40, dist_weight=10.), rpg=dict(weight=dict(reward=2., ent_z=20., ent_a=1.0, mutual_info=1., prior=prior), optim=dict(accumulate_grad=10), use_action_entropy=False), n_batches=200, path='exp/plb/3', max_epochs=40, increasing_reward=1)
    main(z_dim=10, env=dict(low_steps=40, dist_weight=10.), rpg=dict(weight=dict(reward=3., ent_z=20., ent_a=1.0, mutual_info=1., prior=prior), optim=dict(accumulate_grad=10), use_action_entropy=False), n_batches=200, path='exp/plb/4', max_epochs=40, increasing_reward=1) # this roughly work
    #main(z_dim=10, env=dict(low_steps=40, dist_weight=10.), rpg=dict(weight=dict(reward=5., ent_z=20., ent_a=1.0, mutual_info=1., prior=prior), optim=dict(accumulate_grad=10), use_action_entropy=False), n_batches=200, path='exp/plb/5', max_epochs=40, increasing_reward=1)