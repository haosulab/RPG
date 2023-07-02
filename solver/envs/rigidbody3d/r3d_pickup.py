import torch
import nimblephysics as nimble
from solver.envs.rigidbody3d.r3d_grasp import GraspBox
from solver.train_rpg import Trainer
from solver.envs.rigidbody3d.hooks import print_gpu_usage, record_rollout
from solver.envs.rigidbody3d.utils import arr_to_str

# class PickUp(GraspBox):

#     def __init__(self, cfg=None, 
#         low_steps=100, dt=0.005, frame_skip=0, 
#         restitution_coeff=0.0, friction_coeff=1., 
#         n_batches=1, X_OBS_MUL=5., V_OBS_MUL=5., A_ACT_MUL=0.002
#     ):
#         super().__init__(cfg, low_steps, dt, frame_skip, restitution_coeff, friction_coeff, n_batches, X_OBS_MUL, V_OBS_MUL, A_ACT_MUL)

#     def get_obs(self, state=None):
#         return super().get_obs(state).unsqueeze(0).float()

#     def get_action(self, action):
#         return super().get_action(action.squeeze(0))

#     def get_reward(self, s, a, s_next):
#         self.goals = torch.tensor([0, 0.5, 1.5])
#         ee  = nimble.map_to_pos(self.sim.world, self.sim.arm_fk, s) + torch.tensor([0, -0.2, 0])
#         box = self.box_pos(s_next, 0)
#         reach = -(ee - box).norm() ** 2

#         if self.t < 30:
#             gripper_enforce = -(s_next[5] ** 2 + s_next[6] ** 2)
#             reward = (reach + gripper_enforce)
#         elif self.t < 40:
#             gripper_enforce = -((s_next[5] - 0.11) ** 2 + (s_next[6] - 0.11) ** 2)
#             reward = gripper_enforce
#         else:
#             box_to_goal = -(box - self.goals).norm() ** 2
#             reward = (box_to_goal)

#         return reward.unsqueeze(0)


class PickUp(GraspBox):

    def __init__(self, cfg=None):
        super().__init__(cfg)
        # self.X_OBS_MUL = torch.tensor([
        #     3.14, 3.14, 3.14, 0.2, 3.14, 0.2, 0.2,

        # ])
        # self.V_OBS_MUL = V_OBS_MUL
        # self.A_ACT_MUL = A_ACT_MUL
    
    def get_obs(self, state=None):
        obs = super().get_obs(state).unsqueeze(0).float()
        return obs

    def get_action(self, action):
        return super().get_action(action.squeeze(0))

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

    def sample_state_goal(self, batch_size=1):
        state, goals = super().sample_state_goal(batch_size)
        state[3] += 0.2
        goals = torch.tensor([0, 0.5, 1.5])
        return state, goals

    def get_reward(self, s, a, s_next):
        # r_mult = 0.1
        ee  = nimble.map_to_pos(self.sim.world, self.sim.arm_fk, s) \
            + torch.tensor([0, -0.4, 0])
        ee_next = nimble.map_to_pos(self.sim.world, self.sim.arm_fk, s_next) \
            + torch.tensor([0, -0.4, 0])
        box = self.box_pos(s, 0)
        box_next = self.box_pos(s, 0)
        reach = -(ee_next - box_next).norm() ** 2

        gripper_center_deviate = -(s_next[5] - s_next[6]) ** 2 * 2
        gripper_qpos_at_0 = -(s_next[5] ** 2 + s_next[6] ** 2)
        gripper_close = -((s_next[5] - 0.12) ** 2 + (s_next[6] - 0.12) ** 2)
        slip_penalty = -((ee_next - ee) - (box_next - box)).norm() ** 2
        
        if self.t < 20:
            res = dict(
                reward_reach=reach, 
                reward_gripper_qpos=gripper_qpos_at_0, 
                reward_gripper_dev=gripper_center_deviate,
                reward_slip=0,
                reward_box_to_goal=0
            )
        elif self.t < 40:
            res = dict(
                reward_reach=reach,
                reward_gripper_qpos=gripper_close,
                reward_gripper_dev=gripper_center_deviate,
                reward_slip=slip_penalty,
                reward_box_to_goal=0
            )
        else:
            box_to_goal = -(box - self.goals).norm() ** 2
            res = dict(
                reward_reach=reach * 0.1,
                reward_gripper_qpos=gripper_close,
                reward_gripper_dev=gripper_center_deviate,
                reward_slip=slip_penalty,
                reward_box_to_goal=box_to_goal
            )
        return res


def main():

    trainer = Trainer.parse(

        env=dict(
            TYPE="PickUp", 
            n_batches=1,
            cam_resolution=(800,800),
            A_ACT_MUL=0.01, X_OBS_MUL=2.0, V_OBS_MUL=1.0,
            gravity=-1.0
        ),
        
        actor=dict(
            not_func=False,
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
            optim=dict(accumulate_grad=5)
        ),
        z_dim=1, z_cont_dim=0,
        max_epochs=1000,
        record_gif_per_epoch=1,
        device="cpu",

        # book keeping
        path="exp/pickup/debug",
        log_date=False
    )

    trainer.epoch_hooks.append(print_gpu_usage)
    trainer.epoch_hooks.append(record_rollout)
    trainer.start()


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    main()