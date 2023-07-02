
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

    trajs = []

    with torch.no_grad():
        import tqdm
        for i in tqdm.trange(actor._pi_z.action_space.n):
            traj = []
            #print(i)
            z = totensor([i], device='cuda:0', dtype=torch.long)
            obs = env.reset()
            prior_z = trainer.rpg.prior_actor.pi_z(obs, None, timestep=0).sample()[0] 

            p_z = actor.pi_z(obs, prior_z, timestep=0)
            traj.append(env.env.get_state(env.env.get_cur_steps()))
            print(z, p_z.log_prob(z))

            tmp = []
            for t in range(env.low_steps):
                p_a = actor.pi_a(obs, z, timestep=t)
                traj.append(env.env.get_state(env.env.get_cur_steps()))

                a, log_p_a = p_a.rsample()
                print(a, z)
                obs, r, _, _ = env.step(a)
                tmp.append(env.render(mode='rgb_array', render_plt=True, spp=1))
            
            images.append(tmp)
            trajs.append(traj)

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
        logger.torch_save(trajs, f'traj{epoch_id}.th')


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
        env=dict(TYPE='CutEnv', low_steps=40),
        actor=dict(
            not_func=False,
            a_head=dict(
                TYPE='Normal',
                linear=True, # believe in tanh
                squash=False,
                std_mode='fix_no_grad',
                std_scale=0.1
            ),
            ignore_previous_z=True,
        ), # initial 0.3 

        
        softmax=dict(
            decay=0.3,
            decay_epoch=10,
        ),


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
    #base = dict(z_dim=1, rpg=dict(optim=dict(accumulate_grad=0)))
    prior = 0.005
    #main(max_epochs=100, rpg=dict(weight=dict(reward=1., ent_z=0., ent_a=0.0, mutual_info=1., prior=prior)), z_dim=1)
    # main(max_epochs=10, rpg=dict(weight=dict(reward=1., ent_z=0., ent_a=0.0, mutual_info=1., prior=prior)), z_dim=1, path='exp/plb/cut/0')
    # main(max_epochs=10, rpg=dict(weight=dict(reward=1., ent_z=0., ent_a=0.0, mutual_info=1., prior=prior)), z_dim=1, path='exp/plb/cut/1')
    # main(max_epochs=10, rpg=dict(weight=dict(reward=1., ent_z=0., ent_a=0.0, mutual_info=1., prior=prior)), z_dim=1, path='exp/plb/cut/2')
    # main(max_epochs=10, rpg=dict(weight=dict(reward=1., ent_z=0., ent_a=0.0, mutual_info=1., prior=prior)), z_dim=1, path='exp/plb/cut/3')
    # main(max_epochs=100, env=dict(dir=1), rpg=dict(weight=dict(reward=1., ent_z=0., ent_a=0.0, mutual_info=1., prior=prior)), z_dim=1, path='exp/plb/cut2')

    # main(max_epochs=10, rpg=dict(weight=dict(reward=100., ent_z=200., ent_a=1.0, mutual_info=1., prior=prior)), z_dim=10, path='exp/cut_rpg/0', increasing_reward=1, n_batches=500)
    #main(max_epochs=20, rpg=dict(weight=dict(reward=10., ent_z=200., ent_a=1.0, mutual_info=50., prior=prior)), z_dim=10, path='exp/cut_rpg/0', increasing_reward=0, n_batches=50, mutual_decay=0.5, ent_z_decay=0.5)
    # main(max_epochs=20, rpg=dict(weight=dict(reward=10., ent_z=200., ent_a=1.0, mutual_info=10., prior=prior)), z_dim=10, path='exp/cut_rpg/1', increasing_reward=0, n_batches=200, mutual_decay=0.7, ent_z_decay=0.7)

    # main(max_epochs=20, rpg=dict(weight=dict(reward=10., ent_z=200., ent_a=1.0, mutual_info=10., prior=prior)), z_dim=10, path='exp/cut_rpg/2', increasing_reward=0, n_batches=500, mutual_decay=1., ent_z_decay=0.5)
    # main(max_epochs=20, rpg=dict(weight=dict(reward=10., ent_z=200., ent_a=1.0, mutual_info=10., prior=prior), gd=False), z_dim=1, path='exp/cut_pg/0', increasing_reward=0, n_batches=500, mutual_decay=1., ent_z_decay=0.5)
    # main(max_epochs=10, rpg=dict(weight=dict(reward=20., ent_z=200., ent_a=1.0, mutual_info=1., prior=prior)), z_dim=10, path='exp/cut_rpg/softmax/0', increasing_reward=0, n_batches=200, mutual_decay=1., ent_z_decay=1.0, actor=dict(softmax_policy=True))
    # main(max_epochs=10, rpg=dict(weight=dict(reward=10., ent_z=200., ent_a=1.0, mutual_info=1., prior=prior)), z_dim=10, path='exp/cut_rpg/softmax/1', increasing_reward=0, n_batches=200, mutual_decay=1., ent_z_decay=1.0, actor=dict(softmax_policy=True))
    # main(max_epochs=10, rpg=dict(weight=dict(reward=1., ent_z=200., ent_a=1.0, mutual_info=1., prior=prior)), z_dim=10, path='exp/cut_rpg/softmax/2', increasing_reward=0, n_batches=200, mutual_decay=1., ent_z_decay=1.0, actor=dict(softmax_policy=True))
    # main(max_epochs=10, rpg=dict(weight=dict(reward=5., ent_z=200., ent_a=1.0, mutual_info=10., prior=prior)), z_dim=10, path='exp/cut_rpg/softmax/6', increasing_reward=0, n_batches=200, mutual_decay=1., ent_z_decay=1.0, actor=dict(softmax_policy=True))
    # main(max_epochs=10, rpg=dict(weight=dict(reward=1., ent_z=200., ent_a=1.0, mutual_info=1., prior=prior)), z_dim=10, path='exp/cut_rpg/softmax/7', increasing_reward=0, n_batches=200, mutual_decay=0.9, ent_z_decay=1.0, actor=dict(softmax_policy=True))
    # main(max_epochs=10, rpg=dict(weight=dict(reward=1., ent_z=200., ent_a=1.0, mutual_info=1., prior=prior)), z_dim=1, path='exp/cut_rpg/softmax/gd', increasing_reward=0, n_batches=200, mutual_decay=0.9, ent_z_decay=1.0, actor=dict(softmax_policy=True))
    main(max_epochs=10, rpg=dict(weight=dict(reward=1., ent_z=200., ent_a=1.0, mutual_info=1., prior=prior)), z_dim=1, path='exp/cut_rpg/softmax/gd', increasing_reward=0, n_batches=200, mutual_decay=0.9, ent_z_decay=1.0, actor=dict(softmax_policy=True))