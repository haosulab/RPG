import os
import code
import time
import torch
from collections import namedtuple
import nimblephysics as nimble
from solver.envs.rigidbody3d.sapien_viewer import SapienViewer
from sapien.core import Pose
import numpy as np
import cv2
import gym.spaces
import transforms3d.euler
import transforms3d.quaternions
from solver.envs.rigidbody3d.utils import exp_coord2angle_axis, write_text


Actor = namedtuple('Actor', ['urdf', 'nimble_actor', 'sapien_actor'])


class Rigid3dSim:

    def __init__(self, dt=0.001, frame_skip=0,
        resolution=(1366, 1024), distance=3, gravity=-9.81
    ) -> None:
        # nimble simulation world
        self.world = nimble.simulation.World()
        self.world.setGravity([0, gravity, 0])
        self.world.setTimeStep(dt)
        # sapien viewer
        self.viewer = SapienViewer(resolution, distance)
        self.resolution = resolution
        # constants
        self.dt = dt
        self.frame_skip = frame_skip
        self.actors = []
        self.state = None
        # self.enforce_limit = lambda world, x: x

    def load_urdf(self, urdf_path, restitution_coeff=1, friction_coeff=0, mass=None, inertia=None):
        ASSETDIR = (os.path.join(os.path.split(os.path.realpath(__file__))[0], "assets"))
        urdf_path = f"{ASSETDIR}/{urdf_path}"
        # sapien load
        sapien_actor = self.viewer.load_urdf(urdf_path)
        # print("sapien load successful")
        # nimble load
        nimble_actor = self.world.loadSkeleton(urdf_path)
        for bodyNode in nimble_actor.getBodyNodes():
            bodyNode.setRestitutionCoeff(restitution_coeff)
            bodyNode.setFrictionCoeff(friction_coeff)
            if mass is not None:
                # nimble mass is 1 by default
                bodyNode.setMass(mass)
            if inertia is not None:
                bodyNode.setMomentOfInertia(*inertia)
        # save
        actor = Actor(urdf_path, nimble_actor, sapien_actor)
        self.actors.append(actor)
        return actor

    def update_viewer(self):
        # self.viewer.scene.step()
        for actor in self.actors:
            pos = actor.nimble_actor.getPositions()
            if actor.sapien_actor.dof > 0:
                # handle nimble load arm
                actor.sapien_actor.set_qpos(pos)
            elif len(pos) > 0:
                # handle nimble load robot with no joint
                initpose = Pose(
                        p=[0, 0, 0], 
                        q=transforms3d.euler.euler2quat(
                            np.pi / 2, 0, -np.pi / 2
                        )
                )
                theta, omega = exp_coord2angle_axis(pos[:3])
                inputpose = Pose(
                    p=pos[3:], 
                    q=transforms3d.quaternions.axangle2quat(
                        vector=omega, theta=theta
                    )
                )
                actor.sapien_actor.set_pose(
                    initpose.transform(inputpose)
                )
        self.viewer.scene.update_render()
        if self.viewer.window is not None:
            self.viewer.window.render()

    def nimble_loop(self, states):
        gui = nimble.NimbleGUI(self.world)
        gui.serve(8080)  # host the GUI on localhost:8080
        gui.loopStates(states)  # tells the GUI to animate our list of states
        gui.blockWhileServing()  # block here so we don't exit the program

    def sapien_loop(self):
        """ hold sapien window open """
        while not self.viewer.window.closed:
            self.update_viewer()

    def get_init_state(self):
        return torch.zeros(self.world.getStateSize())

    def set_init_state(self, state):
        self.state = state
        self.world.setState(self.state.detach().cpu().numpy())
        # print("set_init_state", self.state)

    def reset(self):
        init_state = self.get_init_state()
        self.set_init_state(init_state)
        return self.state

    def get_reward(self, state, action, next_state):
        return 0

    def get_assist(self):
        return 0
        # return torch.tensor(
        #     self.arm.nimble_actor.getInverseDynamics(
        #         torch.zeros(self.world.getActionSize())
        #     )
        # )

    def step(self, action):
        assist = self.get_assist()
        next_state = nimble.timestep(self.world, self.state, action + assist)
        # next_state = self.enforce_limit(self.world, next_state)
        for _ in range(self.frame_skip):
            assist = self.get_assist()
            next_state = nimble.timestep(self.world, next_state, action + assist)
            # next_state = self.enforce_limit(self.world, next_state)
        reward = self.get_reward(self.state, action, next_state)
        self.state = next_state
        return self.state, reward, False, None

    def render(self, text=None):
        self.update_viewer()
        img = self.viewer.take_picture()[:, :, :3]
        return write_text(img, text, fontsize=int(24/1024*self.resolution[1]))

    @property
    def action_space(self):
        return gym.spaces.Box(-1, 1, (self.world.getActionSize(), ))

    @property
    def observation_space(self):
        return gym.spaces.Box(-1, 1, (self.world.getStateSpace(), ))