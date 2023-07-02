import torch
import gym
import numpy as np
import nimblephysics as nimble
from solver.envs.rigidbody3d.r3d_pickup import PickUp
from solver.train_rpg import Trainer
from solver.envs.rigidbody3d.hooks import print_gpu_usage, record_rollout
from solver.envs.rigidbody3d.utils import arr_to_str


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
        #print(pad_left.shape, pad_right.shape, ee.shape, balls.shape)
        #exit(0)
        # print(f"save_traj: traj saved to {logger.get_dir()}/trajs/{epoch_id}")


class Pick3Cube(PickUp):

    def __init__(self, cfg=None, pad_dir_weight=0.75, write_text=False, boxid=None, reach_reward=1., norm='exp_l2', restitution_coeff=0.1, low_steps=120):
        super().__init__(
            cfg, 
            dt=0.005, frame_skip=0, gravity=-1.0,
            cam_resolution=(512, 512),
            A_ACT_MUL=0.01, X_OBS_MUL=2.0, V_OBS_MUL=1.0
        )
        self.pad_dir_weight = pad_dir_weight
        MULT = 1.
        self.A_ACT_MUL = torch.tensor(
            [1 / 200 /MULT, 1 / 150 /MULT, 1 / 100 /MULT, 1 / 100, 1 / 100, 1 / 100, 1 / 100]
            # [1 / 200, 1 / 200, 1 / 200, 1 / 100, 1 / 200, 1 / 100, 1 / 100]
        )

        self.observation_space = gym.spaces.Box(-1, 1, (self.sim.world.getStateSize()+1, ))

    def init_simulator(self):
        # arm
        self.sim.arm = self.sim.load_urdf(
            "gripper_v2.urdf", 
            restitution_coeff=self.restitution_coeff,
            friction_coeff=self.friction_coeff,
            mass=1, inertia=[0.1, 0.1, 0.1])
        
        # boxes
        self.sim.box = [
            self.sim.load_urdf(
                "sphere.urdf", 
                restitution_coeff=self.restitution_coeff,
                friction_coeff=self.friction_coeff,
                mass=0.1, inertia=[0.1, 0.1, 0.1]),
            self.sim.load_urdf(
                "sphere.urdf", 
                restitution_coeff=self.restitution_coeff,
                friction_coeff=self.friction_coeff,
                mass=0.1, inertia=[0.1, 0.1, 0.1]),
            self.sim.load_urdf(
                "sphere.urdf", 
                restitution_coeff=self.restitution_coeff,
                friction_coeff=self.friction_coeff,
                mass=0.1, inertia=[0.1, 0.1, 0.1])
        ]

        # ground
        self.sim.load_urdf(
            "ground.urdf", 
            restitution_coeff=self.restitution_coeff,
            friction_coeff=self.friction_coeff)
        
        # action only control arm, not other objects
        for i in range(self.sim.arm.sapien_actor.dof, self.sim.world.getActionSize()):
            self.sim.world.removeDofFromActionSpace(i)
        
        # nimble arm forward kinematics
        self.sim.arm_fk = nimble.neural.IKMapping(self.sim.world)
        self.sim.arm_fk.addLinearBodyNode(self.sim.arm.nimble_actor.getBodyNode("end_effector"))

        # nimble pad forward kinematics
        self.sim.pad_fk = nimble.neural.IKMapping(self.sim.world)
        self.sim.pad_fk.addLinearBodyNode(self.sim.arm.nimble_actor.getBodyNode("left_pad"))
        self.sim.pad_fk.addLinearBodyNode(self.sim.arm.nimble_actor.getBodyNode("right_pad"))
    
    def sample_state_goal(self, batch_size=1):
        box_x = 0.4
        box_y = (1 - box_x ** 2) ** 0.5
        box_y *= 0.7

        state = torch.tensor(
            [

            # arm qpos
            -0.6, -2.5, 1.8, -0.6, 0, 0, 0,
            0, 0, 0,      0.0, -0.399,   -0.0,#            -1.5,
            0, 0, 0,      -0.8, -0.399,  0.0,# -(1.65 - box_x),
            0, 0, 0,      -0.4, -0.399,   0.5,# -(1.65 - box_x),
            # 0, 0, 0, 0, -0.399,      0,
            # 0, 0, 0, 0, -0.399,  box_y,
            # 0, 0, 0, 0, -0.399, -box_y,
            # velocities
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
        ]
        )
        goals = torch.tensor(
            [
                     0, 0.4, 1.0,
                 box_y, 0.4, 1.0,
                -box_y, 0.4, 1.0,
            ]
        )
        return state, goals
    
    def get_obs(self, state=None):
        obs = super().get_obs(state)
        b, c = obs.shape
        k = obs.new(b, c+1)
        k[:, :-1] = obs
        k[:, -1] = self.t < 60
        return k
        

    def step(self, action):
        from tools.utils import clamp
        #return super().step(clamp(action, -1., 1.))
        return super().step(action)

    def get_reward(self, s, a, s_next):
        # pos of boxes
        boxes_s_next = torch.stack([
            self.box_pos(s_next, i) 
            for i in range(len(self.sim.box))
        ])
        # ee pos
        # ee_s_next = nimble.map_to_pos(self.sim.world, self.sim.arm_fk, s_next) \
        #     + torch.tensor([0, -0.4, 0])
        # end_effector: nimble.map_to_pos(self.sim.world, self.sim.arm_fk, s_next).view(-1, 3)
        # gripper: nimble.map_to_pos(self.sim.world, self.sim.pad_fk, s_next).view(-1, 3)
        state_v = s_next.reshape(2, -1)
        up_v = state_v[1][3]
        ee_s_next = nimble.map_to_pos(self.sim.world, self.sim.pad_fk, s_next).view(-1, 3).mean(dim=0)\
            + torch.tensor([0, -0.2, 0])
        # select closest box
        # boxi = (ee_s_next - boxes_s_next).norm(dim=1).min(dim=0)[1]
        boxi = self._cfg.boxid
        # self.text += f"\naction: {arr_to_str(a * self.A_ACT_MUL)}"
        # self.text += f"\nselected box: {boxi}"
        # # pad facing
        ee_id = 4
        pad_angle = -(s[:3].sum() - s[ee_id]) - np.pi - (-1.8416)
        # self.text += f"\npad angle: {pad_angle}"
        # self.text += f"\nee pos: {arr_to_str(ee_s_next.detach().numpy())}"
        # rewards
        #reach_top = -(ee_s_next + torch.tensor([0, -0.2, 0]) - box_s_next).norm() ** 2
        #reach = -(ee_s_next - box_s_next).abs().sum()/3 # ** 2
        # gripper_facing = (pad_dir * ref_dir).sum() ** 2
        gripper_center_deviate = -(s_next[ee_id+1] - s_next[ee_id+2]) ** 2
        gripper_qpos_at_0 = -(s_next[ee_id + 1] ** 2 + s_next[ee_id + 2] ** 2)
        gripper_close = -((s_next[ee_id + 1] - 0.1) ** 2 + (s_next[ee_id + 2] - 0.1) ** 2)
        gripper_ground_penalty = ((ee_s_next[1] - 0.2 + 0.4) < 0) * (-(ee_s_next[1] - 0.2 + 0.4) ** 2)

        #boxi = 1
        #boxi=0
        def mynorm(x, method=None):
            method = method or self._cfg.norm
            if method == 'l2':
                return -x.norm() #maximize
            elif method == 'l1':
                return -x.abs().mean()
            elif method == 'exp_l2': # maximie
                return torch.exp(-x.norm()/0.2) + mynorm(x, 'l2')
            elif method == 'exp_l1': # maximie
                return torch.exp(-x.norm()/0.2) + mynorm(x, 'l1')
            else:
                raise NotImplementedError

        if self.t < 70:
            boxes_s_next[:, 1] = 0.21

        
        if boxi is None:

            reach = []
            dists = []
            for i in range(3):
            #for i in [boxi]:
                dists.append((boxes_s_next[i] - ee_s_next).norm())
                reach.append(mynorm(boxes_s_next[i] - ee_s_next))
            #reach = -(ee_s_next - box_s_next).norm() # ** 2
            reach = torch.stack(reach).min(axis=0)[0]
            dist = torch.stack(dists).min(axis=0)[0]
            boxi = 2
        else:
            dist = (boxes_s_next[boxi] - ee_s_next).norm()
            reach = mynorm(boxes_s_next[boxi] - ee_s_next)

        box_s_next = boxes_s_next[boxi]
        goal = self.goals.view(-1, 3)[boxi]

        if self.t < 70:
            res = dict(
                height = -torch.relu(-(ee_s_next[1] - 0.2)) * 2000, #- torch.log(torch.relu(ee_s_next[1] - 0.2)/0.01) # log barrier,
                reward_gripper_center_deviate=gripper_center_deviate * 20,
                reward_gripper_qpos=gripper_qpos_at_0 * 20,
                gripper_ground_penalty=gripper_ground_penalty * 10,
                reward_reach=reach * self._cfg.reach_reward,
            )
            if self.t == 66:
                print(self.t, ee_s_next, res)
        # elif self.t < 50:
        #     res = dict(
        #         reward_reach=reach * self._cfg.reach_reward,
        #         # reward_gripper_facing=-(pad_angle ** 2 * self.pad_dir_weight),#gripper_facing * self.pad_dir_weight, # 0.75 is good
        #         reward_gripper_center_deviate=gripper_center_deviate * 20,
        #         reward_gripper_qpos=gripper_qpos_at_0 * 20,
        #         gripper_ground_penalty=gripper_ground_penalty * 10,
        #     )
        #     if self.t == 49:
        #         print('49', ee_s_next, res)
        else:
            res = dict(
                reward_reach=reach * 1 * self._cfg.reach_reward,
                reward_box_pull=(box_s_next[1] - (-0.42)) * 10, #* torch.exp(-dist/0.06), #-((box_s_next - goal) ** 2).sum() * 10,
                # reward_gripper_facing=-(pad_angle ** 2 * self.pad_dir_weight),
                reward_gripper_center_deviate=gripper_center_deviate * 20,
                reward_gripper_qpos=gripper_close * 20,
                gripper_ground_penalty=gripper_ground_penalty * 10,

                #reward_up = torch.relu(5.-up_v) * 1. if self.t > 100 else 0.,
                reward_up = (ee_s_next[1] - 0.3).clamp(-100, 0.) * 0.01,
            )
            if self.t == self._cfg.low_steps-1:
                print(self._cfg.low_steps-1, ee_s_next, res, up_v)
        # if self.t == 49 or self.t == 99:
        #     print("here")
        #     print(res)
        return res


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
    #main(rpg=dict(weight=dict(ent_z=0., ent_a=0., prior=0.)), env=dict(boxid=0))
    #main(rpg=dict(weight=dict(ent_z=0., ent_a=0., prior=0.)), env=dict(boxid=1))
    #main(rpg=dict(weight=dict(ent_z=20., ent_a=1., prior=0.000001), optim=dict(accumulate_grad=5)), env=dict(boxid=1), z_dim=1, max_epochs=20, path='exp/aseq')
    #main(rpg=dict(weight=dict(ent_z=20., ent_a=1., prior=0.000001), optim=dict(accumulate_grad=5)), env=dict(boxid=1), z_dim=10, max_epochs=20, path='exp/zdim10')
    #main(rpg=dict(weight=dict(ent_z=20., ent_a=1., prior=0.000001), optim=dict(accumulate_grad=5)), env=dict(boxid=1), z_dim=1, max_epochs=20, path='exp/zdim1')
    # increasing reward = 1
    #main(rpg=dict(weight=dict(ent_z=20., ent_a=1., prior=0.000001, reward=100), optim=dict(accumulate_grad=5)), env=dict(boxid=1), increasing_reward=1, z_dim=10, max_epochs=100, exp='exp/aseq1')
    #main(rpg=dict(weight=dict(ent_z=20., ent_a=1., prior=0.000001, reward=0.1), optim=dict(accumulate_grad=5)), env=dict(boxid=1), increasing_reward=0, z_dim=1, max_epochs=100, path='exp/grasp_dim0')
    # main(rpg=dict(weight=dict(ent_z=2000., ent_a=1., prior=0.000001, mutual_info=0.1, reward=10.), optim=dict(accumulate_grad=5)), env=dict(boxid=1), increasing_reward=0, z_dim=10, max_epochs=100, path='exp/g/0')
    # main(rpg=dict(weight=dict(ent_z=2000., ent_a=1., prior=0.000001, mutual_info=0.1, reward=1.), optim=dict(accumulate_grad=5)), env=dict(boxid=1), increasing_reward=0, z_dim=3, max_epochs=100, path='exp/g/1')
    # main(rpg=dict(weight=dict(ent_z=2000., ent_a=1., prior=0.000001, mutual_info=0.1, reward=0.1), optim=dict(accumulate_grad=5)), env=dict(boxid=1), increasing_reward=0, z_dim=10, max_epochs=100, path='exp/g/2')
    # main(rpg=dict(weight=dict(ent_z=2000., ent_a=1., prior=0.000001, mutual_info=0.1, reward=5.), optim=dict(accumulate_grad=5)), env=dict(boxid=1), increasing_reward=0, z_dim=10, max_epochs=100, path='exp/g/3')
    #main(rpg=dict(weight=dict(ent_z=20., ent_a=1., prior=0.000001, reward=10.), optim=dict(accumulate_grad=5)), env=dict(boxid=1), increasing_reward=1, z_dim=10, max_epochs=100, path='exp/aseq3')

    # main(rpg=dict(weight=dict(ent_z=200., ent_a=0.1, prior=0.000001, mutual_info=0.0), optim=dict(accumulate_grad=5)), env=dict(boxid=1), z_dim=10, max_epochs=10, path='exp/m/base', save_traj=dict(per_epoch=1, num_iter=5))
    # main(rpg=dict(weight=dict(ent_z=200., ent_a=0.1, prior=0.000001, mutual_info=0.001), optim=dict(accumulate_grad=5)), env=dict(boxid=1), z_dim=10, max_epochs=10, path='exp/m/0', save_traj=dict(per_epoch=1, num_iter=5))
    # main(rpg=dict(weight=dict(ent_z=200., ent_a=0.1, prior=0.000001, mutual_info=0.01), optim=dict(accumulate_grad=5)), env=dict(boxid=1), z_dim=10, max_epochs=10, path='exp/m/1', save_traj=dict(per_epoch=1, num_iter=5))
    # main(rpg=dict(weight=dict(ent_z=200., ent_a=0.1, prior=0.000001, mutual_info=0.1), optim=dict(accumulate_grad=5)), env=dict(boxid=1), z_dim=10, max_epochs=10, path='exp/m/2', save_traj=dict(per_epoch=1, num_iter=5))
    # main(rpg=dict(weight=dict(ent_z=200., ent_a=0.1, prior=0.000001, mutual_info=1.), optim=dict(accumulate_grad=5)), env=dict(boxid=1), z_dim=10, max_epochs=10, path='exp/m/3', save_traj=dict(per_epoch=1, num_iter=5))
    # main(rpg=dict(weight=dict(ent_z=200., ent_a=0.1, prior=0.000001, mutual_info=10.), optim=dict(accumulate_grad=5)), env=dict(boxid=1), z_dim=10, max_epochs=10, path='exp/m/4', save_traj=dict(per_epoch=1, num_iter=5))

    # main(rpg=dict(weight=dict(ent_z=200., ent_a=0.1, prior=0.000001, mutual_info=1.), optim=dict(accumulate_grad=5)), env=dict(boxid=1), z_dim=10, max_epochs=20, path='exp/e/0', save_traj=dict(per_epoch=1, num_iter=10), actor=dict(a_head=dict(std_scale=0.1)))
    # main(rpg=dict(weight=dict(ent_z=200., ent_a=0.1, prior=0.000001, mutual_info=1.), optim=dict(accumulate_grad=5)), env=dict(boxid=1), z_dim=10, max_epochs=20, path='exp/e/1', save_traj=dict(per_epoch=1, num_iter=10), actor=dict(a_head=dict(std_scale=0.2)))

    #main(rpg=dict(weight=dict(ent_z=200., ent_a=0.1, prior=0.000001, mutual_info=1.), optim=dict(accumulate_grad=5)), env=dict(boxid=1), z_dim=10, max_epochs=10, path='exp/p/0', save_traj=dict(per_epoch=1, num_iter=5))
    #main(rpg=dict(stop_pg=True, weight=dict(ent_z=2000., ent_a=0.1, prior=0.000001, mutual_info=1.), optim=dict(accumulate_grad=5)), env=dict(boxid=1), z_dim=10, max_epochs=10, path='exp/p/3', save_traj=dict(per_epoch=1, num_iter=5))
    #main(rpg=dict(stop_pg=True, weight=dict(ent_z=2000., ent_a=0.1, prior=0.000001, mutual_info=4.), optim=dict(accumulate_grad=5)), env=dict(boxid=1), z_dim=10, max_epochs=10, path='exp/p/4', save_traj=dict(per_epoch=1, num_iter=5))
    #main(rpg=dict(stop_pg=True, weight=dict(ent_z=2000., ent_a=0.1, prior=0.000001, mutual_info=10.), optim=dict(accumulate_grad=5)), env=dict(boxid=1), z_dim=10, max_epochs=10, path='exp/p/5', save_traj=dict(per_epoch=1, num_iter=5))
    # main(rpg=dict(stop_pg=True, weight=dict(ent_z=2000., ent_a=0.1, prior=0.000001, mutual_info=10.), optim=dict(accumulate_grad=5)), env=dict(boxid=1), actor=dict(a_head=dict(std_scale=0.1)), z_dim=10, max_epochs=10, path='exp/p/6', save_traj=dict(per_epoch=1, num_iter=5))
    # main(rpg=dict(stop_pg=True, weight=dict(ent_z=2000., ent_a=0.1, prior=0.000001, mutual_info=10.), optim=dict(accumulate_grad=5)), env=dict(boxid=1), actor=dict(a_head=dict(std_scale=0.2)), z_dim=10, max_epochs=10, path='exp/p/7', save_traj=dict(per_epoch=1, num_iter=5))
    # main(rpg=dict(stop_pg=True, weight=dict(ent_z=2000., ent_a=0.1, prior=0.000001, mutual_info=10.), optim=dict(accumulate_grad=5)), env=dict(boxid=1), actor=dict(a_head=dict(std_scale=0.05)), z_dim=10, max_epochs=10, path='exp/p/8', save_traj=dict(per_epoch=1, num_iter=5))
    # main(rpg=dict(stop_pg=True, weight=dict(ent_z=2000., ent_a=0.1, prior=0.000001, mutual_info=1., reward=0.), optim=dict(accumulate_grad=5)), env=dict(boxid=1), actor=dict(a_head=dict(std_scale=0.1)), z_dim=10, max_epochs=10, path='exp/s/0', save_traj=dict(per_epoch=1, num_iter=20))
    # main(rpg=dict(stop_pg=True, weight=dict(ent_z=2000., ent_a=0.1, prior=0.000001, mutual_info=1.), optim=dict(accumulate_grad=5)), env=dict(boxid=1), actor=dict(a_head=dict(std_scale=0.2)), z_dim=10, max_epochs=10, path='exp/s/1', save_traj=dict(per_epoch=1, num_iter=20))
    # main(rpg=dict(stop_pg=True, weight=dict(ent_z=2000., ent_a=0.1, prior=0.000001, mutual_info=10.), optim=dict(accumulate_grad=5)), env=dict(boxid=1), actor=dict(a_head=dict(std_scale=0.05)), z_dim=10, max_epochs=10, path='exp/p/8', save_traj=dict(per_epoch=1, num_iter=10))

    # main(rpg=dict(stop_pg=True, weight=dict(ent_z=2000., ent_a=0.1, prior=0.000001, mutual_info=1., reward=1.), optim=dict(accumulate_grad=5)), env=dict(boxid=1, reach_reward=0.), actor=dict(a_head=dict(std_scale=0.1)), z_dim=10, max_epochs=10, increasing_reward=0, path='exp/s/1', save_traj=dict(per_epoch=1, num_iter=20))
    # main(rpg=dict(stop_pg=True, weight=dict(ent_z=2000., ent_a=0.1, prior=0.000001, mutual_info=1., reward=1.), optim=dict(accumulate_grad=5)), env=dict(boxid=1, reach_reward=0.01), actor=dict(a_head=dict(std_scale=0.1)), z_dim=10, max_epochs=10, increasing_reward=0, path='exp/s/2', save_traj=dict(per_epoch=1, num_iter=20))
    # main(rpg=dict(stop_pg=True, weight=dict(ent_z=2000., ent_a=0.1, prior=0.000001, mutual_info=1., reward=1.), optim=dict(accumulate_grad=5)), env=dict(boxid=1, reach_reward=1.), actor=dict(a_head=dict(std_scale=0.5)), z_dim=10, max_epochs=10, increasing_reward=0, path='exp/k/0', save_traj=dict(per_epoch=1, num_iter=20))
    # main(rpg=dict(stop_pg=True, weight=dict(ent_z=2000., ent_a=0.1, prior=0.000001, mutual_info=10., reward=1.), optim=dict(accumulate_grad=5)), env=dict(boxid=1, reach_reward=0.1), actor=dict(a_head=dict(std_scale=0.5)), z_dim=10, max_epochs=10, increasing_reward=0, path='exp/k/1', save_traj=dict(per_epoch=1, num_iter=20))
    # main(rpg=dict(stop_pg=True, weight=dict(ent_z=2000., ent_a=0.1, prior=0.000001, mutual_info=1., reward=1.), optim=dict(accumulate_grad=5)), env=dict(boxid=1, reach_reward=0.01), actor=dict(a_head=dict(std_scale=0.5)), z_dim=10, max_epochs=10, increasing_reward=0, path='exp/k/2', save_traj=dict(per_epoch=1, num_iter=20))
    #main(rpg=dict(stop_pg=True, weight=dict(ent_z=2000., ent_a=0.1, prior=0.000001, mutual_info=1., reward=10.), optim=dict(accumulate_grad=5)), env=dict(boxid=1, reach_reward=0.01), actor=dict(a_head=dict(std_scale=0.5)), z_dim=10, max_epochs=10, increasing_reward=0, path='exp/k/3', save_traj=dict(per_epoch=1, num_iter=20))
    #main(rpg=dict(stop_pg=True, weight=dict(ent_z=2000., ent_a=0.1, prior=0.000001, mutual_info=1., reward=10.), optim=dict(accumulate_grad=5)), env=dict(boxid=1, reach_reward=1.), actor=dict(a_head=dict(std_scale=0.5)), z_dim=10, max_epochs=10, increasing_reward=0, path='exp/k/4', save_traj=dict(per_epoch=1, num_iter=20))
    # main(rpg=dict(stop_pg=True, weight=dict(ent_z=2000., ent_a=0.0, prior=0.000001, mutual_info=0., reward=10.), optim=dict(accumulate_grad=5)), env=dict(boxid=1, reach_reward=1.), actor=dict(a_head=dict(std_scale=0.5)), z_dim=10, max_epochs=10, increasing_reward=0, path='exp/k/5', save_traj=dict(per_epoch=1, num_iter=20))

    # main(rpg=dict(stop_pg=True, weight=dict(ent_z=2000., ent_a=0.0, prior=0.000001, mutual_info=0., reward=1.), optim=dict(accumulate_grad=1)), env=dict(boxid=1, reach_reward=10.), actor=dict(a_head=dict(std_scale=0.03)), z_dim=10, max_epochs=10, increasing_reward=0, path='exp/k/5', save_traj=dict(per_epoch=1, num_iter=20))
    # main(rpg=dict(stop_pg=True, weight=dict(ent_z=2000., ent_a=0.0, prior=0.000001, mutual_info=0., reward=1.), optim=dict(accumulate_grad=1)), env=dict(boxid=1, reach_reward=10.), actor=dict(a_head=dict(std_scale=0.03)), z_dim=10, max_epochs=10, increasing_reward=0, path='exp/k/6', save_traj=dict(per_epoch=1, num_iter=20))
    #main(rpg=dict(stop_pg=True, weight=dict(ent_z=2000., ent_a=0.0, prior=0.000001, mutual_info=0., reward=10.), optim=dict(accumulate_grad=5, lr=0.01)), env=dict(boxid=1, reach_reward=1.), actor=dict(a_head=dict(std_scale=0.03)), z_dim=10, max_epochs=10, increasing_reward=0, path='exp/k/6', save_traj=dict(per_epoch=1, num_iter=20))
    #main(rpg=dict(stop_pg=True, weight=dict(ent_z=2000., ent_a=0.0, prior=0.000001, mutual_info=1., reward=10.), optim=dict(accumulate_grad=5, lr=0.01)), env=dict(boxid=1, reach_reward=1.), actor=dict(a_head=dict(std_scale=0.03)), z_dim=10, max_epochs=10, increasing_reward=0, path='exp/k/7', save_traj=dict(per_epoch=1, num_iter=20))
    # main(rpg=dict(stop_pg=True, weight=dict(ent_z=2000., ent_a=0.0, prior=0.000001, mutual_info=1., reward=10.), optim=dict(accumulate_grad=5, lr=0.01)), env=dict(boxid=1, reach_reward=1.), actor=dict(a_head=dict(std_scale=0.03)), z_dim=10, max_epochs=10, increasing_reward=0, path='exp/k/7', save_traj=dict(per_epoch=1, num_iter=20))

    # main(rpg=dict(stop_pg=True, weight=dict(ent_z=2000., ent_a=0.0, prior=0.000001, mutual_info=0., reward=1.), optim=dict(accumulate_grad=1)), env=dict(reach_reward=1.), actor=dict(a_head=dict(std_scale=0.03)), z_dim=1, max_epochs=10, increasing_reward=0, path='exp/k/single', save_traj=dict(per_epoch=1, num_iter=20))
    # main(rpg=dict(stop_pg=True, weight=dict(ent_z=200., ent_a=0.0, prior=0.000001, mutual_info=0., reward=1.), optim=dict(accumulate_grad=1)), env=dict(boxid=None, reach_reward=1., norm='exp_l2', low_steps=100), actor=dict(a_head=dict(std_scale=0.03)), z_dim=1, max_epochs=20, increasing_reward=0, path='exp/k/gd', save_traj=dict(per_epoch=1, num_iter=20))
    #main(rpg=dict(stop_pg=True, weight=dict(ent_z=1000., ent_a=0.0, prior=0.000001, mutual_info=0.3, reward=1.), optim=dict(accumulate_grad=1)), env=dict(reach_reward=1.), actor=dict(a_head=dict(std_scale=0.03)), z_dim=10, path='exp/k/rpg_0.3', max_epochs=40, increasing_reward=0, save_traj=dict(per_epoch=1, num_iter=20))
    # main(rpg=dict(stop_pg=True, weight=dict(ent_z=2000., ent_a=0.0, prior=0.000001, mutual_info=0., reward=1.), optim=dict(accumulate_grad=1)), env=dict(boxid=2, reach_reward=1., norm='exp_l2', low_steps=100), actor=dict(a_head=dict(std_scale=0.03)), z_dim=1, max_epochs=20, increasing_reward=0, path='exp/k/single_grasp', save_traj=dict(per_epoch=1, num_iter=20))
    # main(rpg=dict(stop_pg=True, weight=dict(ent_z=1000., ent_a=0.0, prior=0.000001, mutual_info=0.03, reward=1.), optim=dict(accumulate_grad=1)), env=dict(reach_reward=1.), actor=dict(a_head=dict(std_scale=0.03)), z_dim=10, path='exp/k/rpg_0.03', max_epochs=40, increasing_reward=0, save_traj=dict(per_epoch=1, num_iter=20))
    # main(rpg=dict(stop_pg=False, weight=dict(ent_z=1000., ent_a=0.0, prior=0.000001, mutual_info=0.0, reward=1.), optim=dict(accumulate_grad=1, lr=0.01)), env=dict(reach_reward=1.), actor=dict(a_head=dict(std_scale=0.03)), z_dim=4, path='exp/k/rpg_0.0', max_epochs=40, increasing_reward=0, save_traj=dict(per_epoch=1, num_iter=20))
    # main(rpg=dict(stop_pg=False, weight=dict(ent_z=1000., ent_a=0.0, prior=0.000001, mutual_info=0.1, reward=1.), optim=dict(accumulate_grad=1, lr=0.01)), env=dict(reach_reward=1.), actor=dict(a_head=dict(std_scale=0.03)), z_dim=5, path='exp/k/rpg_0.1', max_epochs=40, increasing_reward=0, save_traj=dict(per_epoch=1, num_iter=20))
    # main(rpg=dict(stop_pg=False, weight=dict(ent_z=1000., ent_a=0.0, prior=0.000001, mutual_info=1., reward=1.), optim=dict(accumulate_grad=10, lr=0.01)), env=dict(reach_reward=1.), actor=dict(a_head=dict(std_scale=0.03)), z_dim=5, path='exp/k/rpg_1._batch_10', max_epochs=40, increasing_reward=0, save_traj=dict(per_epoch=1, num_iter=20))
    # main(rpg=dict(stop_pg=True, weight=dict(ent_z=1000., ent_a=0.0, prior=0.000001, mutual_info=1., reward=1.), optim=dict(accumulate_grad=5, lr=0.05)), env=dict(reach_reward=1.), actor=dict(a_head=dict(std_scale=0.01)), z_dim=5, path='exp/k/rpg_1._stoppg_batch_10', max_epochs=40, increasing_reward=0, save_traj=dict(per_epoch=1, num_iter=20))
  
    main(rpg=dict(stop_pg=True, weight=dict(ent_z=2000., ent_a=0.0, prior=0.000001, mutual_info=0., reward=1.), optim=dict(lr=0.005, accumulate_grad=1)), env=dict(reach_reward=1., boxid=2), actor=dict(a_head=dict(std_scale=0.008)), z_dim=1, max_epochs=10, increasing_reward=0, path='exp/k/single_grasp', save_traj=dict(per_epoch=1, num_iter=20), start_pg=20)
    # main(rpg=dict(stop_pg=True, weight=dict(ent_z=2000., ent_a=0.0, prior=0.000001, mutual_info=0., reward=1.), optim=dict(lr=0.05, accumulate_grad=1)), env=dict(reach_reward=1., boxid=2), actor=dict(a_head=dict(std_scale=0.008)), z_dim=1, max_epochs=10, increasing_reward=0, path='exp/k/single_grasp', save_traj=dict(per_epoch=1, num_iter=20))
    # main(rpg=dict(stop_pg=True, weight=dict(ent_z=1000., ent_a=0.0, prior=0.000001, mutual_info=1., reward=1.), optim=dict(accumulate_grad=5, lr=0.005)), env=dict(reach_reward=1.), actor=dict(a_head=dict(std_scale=0.008), not_func=True), z_dim=5, path='exp/k/rpg_1._not_func', max_epochs=40, increasing_reward=0, save_traj=dict(per_epoch=1, num_iter=20), mutual_decay=0.1)