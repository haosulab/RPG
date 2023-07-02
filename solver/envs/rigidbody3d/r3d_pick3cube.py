import torch
import numpy as np
import nimblephysics as nimble
from solver.envs.rigidbody3d.r3d_pickup import PickUp
from solver.train_rpg import Trainer
from solver.envs.rigidbody3d.hooks import print_gpu_usage, record_rollout
from solver.envs.rigidbody2d.hooks import save_traj
from solver.envs.rigidbody3d.utils import arr_to_str


class Pick3Cube(PickUp):

    def __init__(self, cfg=None, pad_dir_weight=0.75):
        super().__init__(
            cfg, 
            dt=0.005, frame_skip=0, gravity=-1.0,
            cam_resolution=(512, 512),
            A_ACT_MUL=0.01, X_OBS_MUL=2.0, V_OBS_MUL=1.0
        )
        self.pad_dir_weight = pad_dir_weight
        self.A_ACT_MUL = torch.tensor(
            [1 / 200, 1 / 150, 1 / 100, 1 / 100, 1 / 100, 1 / 100, 1 / 100]
            # [1 / 200, 1 / 200, 1 / 200, 1 / 100, 1 / 200, 1 / 100, 1 / 100]
        )

    def init_simulator(self):
        # arm
        self.sim.arm = self.sim.load_urdf(
            "gripper_v2.urdf", 
            restitution_coeff=self.restitution_coeff,
            friction_coeff=self.friction_coeff,
            mass=1)
        
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
        state = torch.tensor(
            [
                # arm qpos
                -1.57 / 2 - 0.2, 0.7, -0.7, -0.2, 2.22, 0, 0,
                0, 0, 0,      0, -0.399, 1.0,#            -1.5,
                0, 0, 0,  box_y, -0.399, 1.0,# -(1.65 - box_x),
                0, 0, 0, -box_y, -0.399, 1.0,# -(1.65 - box_x),
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

    def get_reward(self, s, a, s_next):
        # pos of boxes
        boxes_s_next = torch.stack([
            self.box_pos(s_next, i) 
            for i in range(len(self.sim.box))
        ])
        # ee pos
        # ee_s_next = nimble.map_to_pos(self.sim.world, self.sim.arm_fk, s_next) \
        #     + torch.tensor([0, -0.4, 0])
        ee_s_next = nimble.map_to_pos(self.sim.world, self.sim.pad_fk, s_next).view(-1, 3).mean(dim=0)\
            + torch.tensor([0, -0.2, 0])
        # select closest box
        # boxi = (ee_s_next - boxes_s_next).norm(dim=1).min(dim=0)[1]
        boxi = 0
        self.text += f"\naction: {arr_to_str(a * self.A_ACT_MUL)}"
        self.text += f"\nselected box: {boxi}"
        box_s_next = boxes_s_next[boxi]
        goal = self.goals.view(-1, 3)[boxi]
        # pad facing
        pad_angle = -(s[:3].sum() - s[4]) - np.pi
        self.text += f"\npad angle: {pad_angle}"
        self.text += f"\nee pos: {arr_to_str(ee_s_next.detach().numpy())}"
        # rewards
        reach_top = -(ee_s_next + torch.tensor([0, -0.2, 0]) - box_s_next).norm() ** 2
        reach = -(ee_s_next - box_s_next).norm() ** 2
        # gripper_facing = (pad_dir * ref_dir).sum() ** 2
        gripper_center_deviate = -(s_next[5] - s_next[6]) ** 2
        gripper_qpos_at_0 = -(s_next[5] ** 2 + s_next[6] ** 2)
        gripper_close = -((s_next[5] - 0.12) ** 2 + (s_next[6] - 0.12) ** 2)
        gripper_ground_penalty = ((ee_s_next[1] - 0.2 + 0.4) < 0) * (-(ee_s_next[1] - 0.2 + 0.4) ** 2)

        if self.t < 50:
            res = dict(
                reward_reach=reach,
                reward_gripper_facing=-(pad_angle ** 2 * self.pad_dir_weight),#gripper_facing * self.pad_dir_weight, # 0.75 is good
                reward_gripper_center_deviate=gripper_center_deviate * 10,
                reward_gripper_qpos=gripper_qpos_at_0 * 10,
                gripper_ground_penalty=gripper_ground_penalty * 10,
            )
        else:
            res = dict(
                reward_reach=reach,
                reward_box_pull=-((box_s_next - goal) ** 2).sum(),
                reward_gripper_facing=-(pad_angle ** 2 * self.pad_dir_weight),
                reward_gripper_center_deviate=gripper_center_deviate * 10,
                reward_gripper_qpos=gripper_close * 10,
                gripper_ground_penalty=gripper_ground_penalty * 10,
            )
        if self.t == 49 or self.t == 99:
            print("here")
            print(res)
        return res


def main():

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
                std_scale=0.1)
        ),
        
        # RPG
        rpg=dict(
            gd=True,
            optim=dict(
                accumulate_grad=5)
        ),

        z_dim=1, z_cont_dim=0,
        max_epochs=1000, n_batches=100,
        record_gif_per_epoch=1,
        device="cpu",

        # book keeping
        path="exp/aseq",
        log_date=True
    )

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
    main()

