import code
import gym
import time
import torch
from collections import namedtuple
import nimblephysics as nimble

from sapien.core import Pose
import numpy as np
import cv2
import gym.spaces
import transforms3d.euler
import transforms3d.quaternions

from solver.envs.rigidbody3d.sapien_viewer import SapienViewer
from solver.envs.rigidbody3d.utils import exp_coord2angle_axis, write_text
from solver.envs.rigidbody3d.rigid3d_simulator import Rigid3dSim
from solver.envs.goal_env_base import GoalEnv


class Rigid3dBase(GoalEnv):

    def __init__(self, cfg=None, 
        low_steps=200, dt=0.005, frame_skip=0, restitution_coeff=1., friction_coeff=0., gravity=-9.81,
        n_batches=1, X_OBS_MUL=1., V_OBS_MUL=1., A_ACT_MUL=1.,
        cam_resolution=(1366, 1024), cam_distance=3, write_text=True,
    ):
        super().__init__(cfg, num_stages=1, low_steps=low_steps)

        # constants
        self.X_OBS_MUL = X_OBS_MUL
        self.V_OBS_MUL = V_OBS_MUL
        self.A_ACT_MUL = A_ACT_MUL
        self.dt = dt
        self.frame_skip = frame_skip
        self.friction_coeff = friction_coeff
        self.restitution_coeff = restitution_coeff
        self.n_batches = n_batches

        # simulator
        self.sim = Rigid3dSim(dt, frame_skip, cam_resolution, cam_distance, gravity=gravity)
        self.init_simulator()

        self.observation_space = gym.spaces.Box(-1, 1, (self.sim.world.getStateSize(), ))
        self.action_space = gym.spaces.Box(-1, 1, (self.sim.world.getActionSize(), ))
        self.eval_mode = False
        self.goals = None
        self.t = 0

        self.text = ""

    def train(self):
        self.eval_mode = False

    def eval(self):
        self.eval_mode = True

    @property
    def batch_size(self):
        return self.n_batches

    def reset_state_goal(self, states, goals):
        self.sim.set_init_state(states)
        self.goals = goals
        self.t = 0

    def get_obs(self, state=None):
        if state is None:
            state = self.sim.state
        state_size = state.shape[-1]
        return torch.cat([
            state[:state_size // 2] / self.X_OBS_MUL,
            state[state_size // 2:] / self.V_OBS_MUL
        ], dim=-1)

    def get_state(self):
        return self.sim.state.clone()

    def get_action(self, action):
        return action / self.A_ACT_MUL

    def step(self, action, text=None):

        s = self.sim.state
        a = self.get_action(action)
        s_, _, _, _ = self.sim.step(a)
        r = self.get_reward(s, a, s_)
        obs_ = self.get_obs(s_)

        info = dict()
        # save & log 1d obs & action value
        obs1d = obs_.squeeze()
        for i in range(len(obs1d)):
            info[f"obs_{i}"] = obs1d[i].item()
        act1d = action.squeeze()
        for i in range(len(act1d)):
            info[f"act_{i}"] = act1d[i].item()
        # save & log reward decomposition
        if isinstance(r, dict):
            sum_of_reward_components = 0
            for k, v in r.items():
                sum_of_reward_components = \
                    sum_of_reward_components + v
                if isinstance(v, torch.Tensor):
                    v = v.item()
                info[k] = v
            # n_batch = 1
            r = sum_of_reward_components.unsqueeze(0)
            assert len(r.shape) == 1, f"got shape {r.shape}"
        # save render image
        # if text is not None:
        #     info["img"] = self.sim.render(text)
        if self.eval_mode:
            if text is None: text = self.text
            text = f"T:{self.t}\n" + text
            if (not self._cfg.write_text):
                text = None
            info["img"] = self.sim.render(text)
            self.text = ""

        self.t += 1
        return obs_, r, False, info

    def _render_rgb(self, mode="rgb_array", text=None):
        if text is not None:
            text = text + self.text
        if not self._cfg.write_text:
            text = None
        img = self.sim.render(text=text)
        self.text = ""
        return img

    def _render_traj_rgb(self, states, **kwargs):
        # TODO: save_pcb
        return self.sim.render()

    """ methods to be implemented in subclasses """
    """ below is a working example """
        
    def init_simulator(self):
        self.sim.arm    = self.sim.load_urdf("beginner.urdf", friction_coeff=self.friction_coeff)
        self.sim.ball   = self.sim.load_urdf("sphere.urdf", friction_coeff=self.friction_coeff)
        self.sim.ground = self.sim.load_urdf("ground.urdf", friction_coeff=self.friction_coeff)
        # action only control arm, not other objects
        for i in range(self.sim.arm.sapien_actor.dof, self.sim.world.getActionSize()):
            self.sim.world.removeDofFromActionSpace(i)
        # nimble forward kinematics
        self.sim.arm_fk = nimble.neural.IKMapping(self.sim.world)
        self.sim.arm_fk.addLinearBodyNode(self.sim.arm.nimble_actor.getBodyNode("end_effector"))

    def end_effector_pos(self, state):
        return nimble.map_to_pos(self.sim.world, self.sim.arm_fk, state).float()
    
    def end_effector_vel(self, state):
        return nimble.map_to_vel(self.sim.world, self.sim.arm_fk, state).float()

    def sample_state_goal(self, batch_size=1):
        state = torch.tensor(
            [   
                # arm qpos
                -0.2, 0.4, 0, 0.2,
                # ball pos: [expcoord, xyz]
                0, 0, 0, 0, 0, 1,
                # velocities
                0, 0, 0, 0,
                0, 0, 0, 0, 0, 0
            ], dtype=torch.float32
        )
        goals = torch.tensor([0, 0, -3], dtype=torch.float32)
        return state, goals
    
    def get_reward(self, s, a, s_next):
        return -(self.end_effector_pos(self.sim.state) - self.sim.state[7:10]).norm()


if __name__ == "__main__":
    import cv2
    env = Rigid3dBase()
    env.reset()
    cv2.imwrite("debug.png", env.render())
    env.sim.viewer.create_window()
    env.sim.sapien_loop()