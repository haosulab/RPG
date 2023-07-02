import code
import os
import numpy as np
import cv2
import gym
import torch
from tools.utils import animate
import tqdm
import time
from .r3d_base import Rigid3dBase
import nimblephysics as nimble
from .utils import arr_to_str

import os



class GraspBox(Rigid3dBase):

    def __init__(self, cfg=None, 
        low_steps=100, dt=0.005, frame_skip=0, 
        restitution_coeff=0.1, friction_coeff=1., n_batches=1, 
        X_OBS_MUL=5.0, V_OBS_MUL=5.0, A_ACT_MUL=0.0002
    ):
        super().__init__(cfg) # , low_steps, dt, frame_skip, restitution_coeff, friction_coeff, n_batches, X_OBS_MUL, V_OBS_MUL, A_ACT_MUL
        self.action_space = gym.spaces.Box(-1, 1, (self.sim.world.getActionSize(), ))  # gripper

    def init_simulator(self):

        # arm
        self.sim.arm    = self.sim.load_urdf(
            "gripper_v2.urdf", 
            restitution_coeff=self.restitution_coeff,
            friction_coeff=self.friction_coeff,
            mass=1)
        
        # boxes
        self.sim.box = [
            self.sim.load_urdf(
                "box.urdf", 
                restitution_coeff=self.restitution_coeff,
                friction_coeff=self.friction_coeff,
                mass=1, inertia=[0.1, 0.1, 0.1]),
            # self.sim.load_urdf(
            #     "./assets/box.urdf", 
            #     restitution_coeff=self.restitution_coeff,
            #     friction_coeff=self.friction_coeff,
            #     mass=1, inertia=[0.1, 0.1, 0.1]),
            # self.sim.load_urdf(
            #     "./assets/box.urdf", 
            #     restitution_coeff=self.restitution_coeff,
            #     friction_coeff=self.friction_coeff,
            #     mass=1, inertia=[0.1, 0.1, 0.1])
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
        # self.sim.arm_fk.addLinearBodyNode(self.sim.arm.nimble_actor.getBodyNode("left_pad"))
        # self.sim.arm_fk.addLinearBodyNode(self.sim.arm.nimble_actor.getBodyNode("right_pad"))

    def get_action(self, action):
        # handle RL gripper action
        # translated_action = torch.zeros(self.action_space.shape[0] + 1, dtype=action.dtype)
        # translated_action[:-1] = action
        # translated_action[-1] = -action[-1]
        # return super().get_action(translated_action)
        return super().get_action(action)  # normalizes action


    def sample_state_goal(self, batch_size=1):
        # set up init pos & qpos & goal
        box_x = 0.4
        box_y = (1 - box_x ** 2) ** 0.5
        state = torch.tensor(
            [   
                # arm qpos
                -1.57 / 2 - 0.2, 0.7, -0.7, -0.2, 2.22, 0, 0,
                # 0, -2.2, 1.79, -0.2, 1.08, 0, 0,
                # box pos: [expcoord, xyz]
                # 0, 1.57 * 0.5, -1.57 * 0.7, 0, 0, 1,
                0, 0, 0,    0, -0.399, 1.5, #0.65,
                # 0, 0, 0,  box_y, -0.399, 1.65 - box_x,
                # 0, 0, 0, -box_y, -0.399, 1.65 - box_x,
                # joint limit box
                # 0, 0, 0, 0, 0, 0,
                # 0, 0, 0, 0, 0, 0,
                # velocities
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
                # 0, 0, 0, 0, 0, 0,
                # 0, 0, 0, 0, 0, 0,
            ], dtype=torch.float32
        )
        goals = torch.tensor([0, 1, -1.5], dtype=torch.float32)
        return state, goals

    def get_reward(self, s, a, s_next):
        # try to reach rhs first, nothing too fancy
        # ee_pos_s  = nimble.map_to_pos(self.sim.world, self.sim.arm_fk, s).float()
        ee_pos_s_next = nimble.map_to_pos(self.sim.world, self.sim.arm_fk, s_next).float()
        reach = -(ee_pos_s_next - self.goals).norm()
        # help enforce gripper pos joint limit
        gripper_enforce = -(s_next[5] ** 2 + s_next[6] ** 2) * 10
        return reach + gripper_enforce

    def box_pos(self, state, box_index):
        assert 0 <= box_index < len(self.sim.box)
        return state[7 + 6 * (box_index) + 3: 7 + 6 * (box_index) + 6].clone()


if __name__ == "__main__":
    import cv2
    env = GraspBox()
    obs = env.reset()
    cv2.imwrite("debug.png", env.render())

    print(env.end_effector_pos(env.sim.state))

    env.sim.viewer.create_window()
    while not env.sim.viewer.window.closed:
        env.step(torch.tensor([0, 0, 0, -1, 0, 0, 0], dtype=torch.float32))
        env.render()

        print(nimble.map_to_pos(env.sim.world, env.sim.arm_fk, env.sim.state).float())
    
    pass