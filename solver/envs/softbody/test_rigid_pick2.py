import torch
import gym
import numpy as np
import nimblephysics as nimble
from solver.envs.rigidbody3d.r3d_pickup import PickUp
from solver.train_rpg import Trainer
from solver.envs.rigidbody3d.hooks import print_gpu_usage, record_rollout
from solver.envs.rigidbody3d.utils import arr_to_str
from solver.envs.softbody.train_separately import Pick3Cube


@torch.no_grad()
def save_traj(trainer: Trainer, mainloop_locals):
    print("----- saving trajs -----")
    epoch_id = mainloop_locals["epoch_id"]
    if trainer._cfg.save_traj.per_epoch > 0 and epoch_id % trainer._cfg.save_traj.per_epoch == 0:
        trajs = []


        points = []
        for _ in range(trainer._cfg.save_traj.num_iter):
            traj = trainer.rpg.inference(trainer.env, return_state=True)
            trajs.append(traj)

            sim = trainer.env.sim
            pad_left = []
            pad_right = []
            end_effector = []
            balls = []
            for s in traj['state']:
                pad_loc = nimble.map_to_pos(sim.world, sim.pad_fk, s).view(-1, 3)
                ee_loc = nimble.map_to_pos(sim.world, sim.arm_fk, s).view(3)

                pad_left.append(pad_loc[0])
                pad_right.append(pad_loc[1])
                end_effector.append(ee_loc)
                balls.append(torch.cat([trainer.env.box_pos(s, i) for i in range(3)], -1))

            pad_left = torch.stack(pad_left)
            pad_right = torch.stack(pad_right)
            ee = torch.stack(end_effector)
            balls = torch.stack(balls)

            points.append([pad_left, pad_right, ee, balls])

        p = []
        for i in range(4):
            k = torch.stack([j[i] for j in points], 1)
            #k[:, 1] += 4.
            p.append(k/2)

        images = []
        for j in range(0, len(p[0]), 5):
            pp = [i[j] for i in p]
            import matplotlib.pyplot as plt
            from tools.utils import plt_save_fig_array, tonumpy
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            for s, c in zip(pp, ['r', 'g', 'b', 'y']):
                s = tonumpy(s)
                s = s.reshape(-1, 3)
                ax.scatter(s[:, 0], s[:, 2], s[:, 1], color=c)

            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')

            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
            ax.set_zlim(-1, 4)
            img = plt_save_fig_array(fig)
            images.append(img)
            plt.close()

        from tools.utils import logger
        logger.animate(images, 'traj.mp4')

        epoch_id = mainloop_locals['epoch_id']

        logger.torch_save(trajs, f'traj{epoch_id}.th')

class Preprocess:
    def __call__(self, s):
        s_new = s.clone()
        #s_new[:, 0, :4] = 0
        #s_new[:, 0, 6:] = 0
        s_new[:, 5:] = 0
        return s_new


def main(**kwargs):

    trainer = Trainer.parse(

        env=dict(
            TYPE="Pick3Cube", 
            n_batches=1,
        ),
        
        actor=dict(
            not_func=True,
            a_head=dict(
                TYPE='Normal', 
                linear=True, 
                squash=False, 
                std_mode='fix_no_grad', 
                std_scale=0.001)
        ),
        
        # RPG
        rpg=dict(
            gd=True,
            optim=dict(
                lr=0.05,
                accumulate_grad=1),
            weight=dict(prior=0.),
        ),

        info_net=dict(action_weight=0.),

        z_dim=1, z_cont_dim=0,
        max_epochs=1000, n_batches=100,
        record_gif_per_epoch=1,
        device="cpu",

        save_traj=dict(
            per_epoch=1,
            num_iter=1
        ),

        # book keeping
        path="exp/aseq",
        log_date=False,
        _update = kwargs,
        increasing_reward=0,
        

        softmax=dict(
            decay=0.3,
            decay_epoch=20,
        ),
    )

    trainer.rpg.info_log_q.preprocess = Preprocess() # hook to preprocess the state

    trainer.epoch_hooks.append(print_gpu_usage)
    trainer.epoch_hooks.append(save_traj)
    trainer.epoch_hooks.append(record_rollout)
    trainer.start()


if __name__ == "__main__":
    
    # import cv2
    # env = Pick3Cube()
    # obs = env.reset()
    # while True:
    #     env.step(torch.randn(1, env.action_space.shape[0]))
    #     env.get_reward(env.sim.state, torch.zeros(env.action_space.shape[0]), env.sim.state)
    #     cv2.imwrite("debug.png", env.render(text=""))
    #     input()

    # env.sim.viewer.create_window()
    # while not env.sim.viewer.window.closed:
    #     # env.step(torch.zeros(env.action_space.shape[0]))
    #     env.render()

    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    # main(rpg=dict(stop_pg=True, weight=dict(ent_z=2000., ent_a=0.0, prior=0.000001, mutual_info=0., reward=1.), optim=dict(lr=0.005, accumulate_grad=1)), env=dict(reach_reward=1., boxid=2), actor=dict(a_head=dict(std_scale=0.008)), z_dim=1, max_epochs=10, increasing_reward=0, path='exp/k/pick2', save_traj=dict(per_epoch=1, num_iter=20))
    # main(rpg=dict(stop_pg=True, weight=dict(ent_z=1000., ent_a=0.0, prior=0.000001, mutual_info=1., reward=1.), optim=dict(accumulate_grad=5, lr=0.005)), env=dict(reach_reward=1.), actor=dict(a_head=dict(std_scale=0.008), not_func=True), z_dim=5, path='exp/k/pick2_rpg_not_func', max_epochs=40, increasing_reward=0, save_traj=dict(per_epoch=1, num_iter=20), mutual_decay=0.9)
    # main(rpg=dict(stop_pg=True, weight=dict(ent_z=1000., ent_a=0.0, prior=0.000001, mutual_info=1., reward=1.), optim=dict(accumulate_grad=5, lr=0.005)), env=dict(reach_reward=1.), actor=dict(a_head=dict(std_scale=0.008), not_func=True), z_dim=5, path='exp/k/pick2_without_pg', max_epochs=40, increasing_reward=0, save_traj=dict(per_epoch=1, num_iter=20), mutual_decay=0.9, start_pg_epoch=20)
    # main(rpg=dict(stop_pg=True, weight=dict(ent_z=1000., ent_a=0.0, prior=0.000001, mutual_info=1., reward=1.), optim=dict(accumulate_grad=5, lr=0.005)), env=dict(reach_reward=1.), actor=dict(a_head=dict(std_scale=0.008), not_func=True), z_dim=1, path='exp/k/gd', max_epochs=40, increasing_reward=0, save_traj=dict(per_epoch=1, num_iter=20), mutual_decay=0.9, start_pg_epoch=20)
    # main(rpg=dict(stop_pg=True, weight=dict(ent_z=1000., ent_a=0.0, prior=0.000001, mutual_info=1., reward=1.), gd=False, optim=dict(accumulate_grad=5, lr=0.005)), env=dict(reach_reward=1.), actor=dict(a_head=dict(std_scale=0.1), not_func=True), z_dim=1, path='exp/k/pg', max_epochs=40, increasing_reward=0, save_traj=dict(per_epoch=1, num_iter=20), mutual_decay=0.9, start_pg_epoch=20)


    # main(rpg=dict(stop_pg=True, weight=dict(ent_z=100., ent_a=0.0, prior=0.000001, mutual_info=1., reward=1.), optim=dict(accumulate_grad=5, lr=0.005)), env=dict(reach_reward=0.2), actor=dict(a_head=dict(std_scale=0.008), not_func=True), z_dim=5, path='exp/k/pick2_without_pg_2', max_epochs=40, increasing_reward=0, save_traj=dict(per_epoch=1, num_iter=20), mutual_decay=0.9, start_pg_epoch=20)
    # main(rpg=dict(stop_pg=True, weight=dict(ent_z=1000., ent_a=0.0, prior=0.000001, mutual_info=1., reward=1.), optim=dict(accumulate_grad=5, lr=0.005)), env=dict(reach_reward=0.2), actor=dict(a_head=dict(std_scale=0.008), not_func=True), z_dim=1, path='exp/k/gd_2', max_epochs=40, increasing_reward=0, save_traj=dict(per_epoch=1, num_iter=20), mutual_decay=0.9, start_pg_epoch=20)
    # main(rpg=dict(stop_pg=True, weight=dict(ent_z=1000., ent_a=0.0, prior=0.000001, mutual_info=1., reward=1.), gd=False, optim=dict(accumulate_grad=5, lr=0.005)), env=dict(reach_reward=0.2), actor=dict(a_head=dict(std_scale=0.1), not_func=True), z_dim=1, path='exp/k/pg_2', max_epochs=40, increasing_reward=0, save_traj=dict(per_epoch=1, num_iter=20), mutual_decay=0.9, start_pg_epoch=20)

    # main(rpg=dict(stop_pg=True, weight=dict(ent_z=1000., ent_a=0.0, prior=0.000001, mutual_info=1., reward=1.), gd=False, optim=dict(accumulate_grad=5, lr=0.005)), env=dict(reach_reward=0.2), actor=dict(a_head=dict(std_scale=0.1), not_func=True), z_dim=1, path='exp/k/pg_3', max_epochs=40, increasing_reward=0, save_traj=dict(per_epoch=1, num_iter=20), mutual_decay=0.9, start_pg_epoch=20)
    # main(rpg=dict(stop_pg=True, weight=dict(ent_z=1000., ent_a=0.0, prior=0.000001, mutual_info=1., reward=1.), gd=True, optim=dict(accumulate_grad=5, lr=0.005)), env=dict(reach_reward=0.2), actor=dict(a_head=dict(std_scale=0.1), not_func=True), z_dim=1, path='exp/k/gd_3', max_epochs=40, increasing_reward=0, save_traj=dict(per_epoch=1, num_iter=20), mutual_decay=0.9, start_pg_epoch=20)

    # main(rpg=dict(stop_pg=True, weight=dict(ent_z=100., ent_a=0.0, prior=0.000001, mutual_info=5., reward=1.), optim=dict(accumulate_grad=1, lr=0.005)), env=dict(reach_reward=0.2), actor=dict(a_head=dict(std_scale=0.008), not_func=True, softmax_policy=True), z_dim=10, path='exp/k/pick2/softmax', max_epochs=40, increasing_reward=0, save_traj=dict(per_epoch=1, num_iter=20), mutual_decay=1.0, ent_z_decay=1.0, start_pg_epoch=30)
    # main(rpg=dict(stop_pg=True, weight=dict(ent_z=100., ent_a=0.0, prior=0.000001, mutual_info=5., reward=1.), optim=dict(accumulate_grad=5, lr=0.005)), env=dict(reach_reward=0.2), actor=dict(a_head=dict(std_scale=0.008), not_func=True, softmax_policy=True), z_dim=10, path='exp/k/pick2/softmax', max_epochs=40, increasing_reward=0, save_traj=dict(per_epoch=1, num_iter=20), mutual_decay=1.0, ent_z_decay=1.0, start_pg_epoch=30)
    # main(rpg=dict(stop_pg=True, weight=dict(ent_z=100., ent_a=0.0, prior=0.000001, mutual_info=5., reward=1.), optim=dict(accumulate_grad=5, lr=0.005)), env=dict(reach_reward=0.2), actor=dict(a_head=dict(std_scale=0.008), not_func=True, softmax_policy=True), z_dim=10, path='exp/k/pick2/plot', max_epochs=40, increasing_reward=0, save_traj=dict(per_epoch=1, num_iter=20), mutual_decay=1.0, ent_z_decay=1.0, start_pg_epoch=30)
    # main(rpg=dict(stop_pg=True, weight=dict(ent_z=100., ent_a=0.0, prior=0.000001, mutual_info=10., reward=1.), optim=dict(accumulate_grad=5, lr=0.005)), env=dict(reach_reward=0.2), actor=dict(a_head=dict(std_scale=0.008), not_func=True, softmax_policy=True), z_dim=10, path='exp/k/pick2/plot_2', max_epochs=40, increasing_reward=0, save_traj=dict(per_epoch=1, num_iter=20), mutual_decay=0.9, ent_z_decay=1.0, start_pg_epoch=30)
    # main(rpg=dict(stop_pg=True, weight=dict(ent_z=100., ent_a=0.0, prior=0.000001, mutual_info=1., reward=1.), optim=dict(accumulate_grad=5, lr=0.005)), env=dict(reach_reward=0.2), actor=dict(a_head=dict(std_scale=0.008), not_func=True, softmax_policy=True), z_dim=10, path='exp/k/pick2/plot_3', max_epochs=40, increasing_reward=0, save_traj=dict(per_epoch=1, num_iter=20), mutual_decay=0.9, ent_z_decay=1.0, start_pg_epoch=30)
    main(rpg=dict(stop_pg=True, weight=dict(ent_z=100., ent_a=0.0, prior=0.000001, mutual_info=1., reward=1.), optim=dict(accumulate_grad=5, lr=0.005)), env=dict(reach_reward=0.2), actor=dict(a_head=dict(std_scale=0.1), not_func=True, softmax_policy=True), z_dim=1, path='exp/k/pick2/gd', max_epochs=40, increasing_reward=0, save_traj=dict(per_epoch=1, num_iter=20), mutual_decay=0.9, ent_z_decay=1.0, start_pg_epoch=30)