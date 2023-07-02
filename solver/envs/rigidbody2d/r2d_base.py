from threading import local
from cv2 import circle
import numpy as np
import gym
from pyrsistent import b
import torch

import solver.envs.rigidbody2d.rigid2d as r2d
from solver.envs.goal_env_base import GoalEnv


class Rigid2dBase(GoalEnv):

    @property
    def batch_size(self):
        return self.n_batches

    def eval(self):
        self.eval_mode = True

    def train(self):
        self.eval_mode = False

    def __init__(
        self, cfg=None, 
        low_steps=200, dt=0.02, frame_skip=1, friction_coeff=8,
        n_batches=32, X_OBS_MUL=5.0, V_OBS_MUL=5.0, A_ACT_MUL=1.0
    ) -> None:
        super().__init__(low_steps=low_steps)

        self.X_OBS_MUL = X_OBS_MUL
        self.V_OBS_MUL = V_OBS_MUL
        self.A_ACT_MUL = A_ACT_MUL

        self.dt = dt
        self.frame_skip = frame_skip
        self.friction_coeff = friction_coeff
        self.circles: r2d.Circles = None
        self.goals = None
        self.n_batches = n_batches

        self.observation_space = gym.spaces.Box(-1, 1, (4 * 2 + 2,))
        self.action_space = gym.spaces.Box(-1, 1, (2,))

        self.t = 0
        self.eval_mode = False

    def reset_state_goal(self, states, goals):
        self.circles = r2d.Circles(states, self.get_colors())
        self.goals = goals
        self.t = 0

    """ To be implemented in child classes """

    def get_obs(self):
        obs = torch.cat([
            self.circles.state[:, :2, 0:2] / self.X_OBS_MUL,
            self.circles.state[:, :2, 2:4] / self.V_OBS_MUL
        ], dim=-1)
        return torch.cat([
            obs.view(self.batch_size, -1),
            self.goals / self.X_OBS_MUL # goals
        ], dim=-1)

    def get_state(self):
        return self.circles.state.clone()

    def get_colors(self):
        if not hasattr(self, "color"):
            self.color = np.array([
                [255,   0,   0],
                [  0,   0, 255],
                [  0,   0,   0],
            ])
        return self.color

    def sample_state_goal(self, batch_size):
        state = r2d.tensor([
            [-3.7,  0, 2, 0, 0.17,   1], # actor
            [-3.1,  0, 0, 0,  0.3,   3], # object
            [   0,  0, 0, 0,  1.5, 1e6], # obstacle
        ], batch_size=self.batch_size)
        goals = r2d.tensor([3, 0], self.batch_size)
        return state, goals

    def get_reward(self, s, a, s_next):
        """ 
            reward of transition s->a->s_next
            for base env, task is push obj to goal
        """
        g = self.goals      # goal
        x = s_next[:, 1, :2]  # object
        t = s_next[:, 0, :2]  # actor

        dist_to_goal  = (x - g).norm(dim=-1)
        dist_to_actor = (x - t).norm(dim=-1)
        r = -(dist_to_goal + dist_to_actor) / (4 * self.X_OBS_MUL)
        return r

    def get_action(self, action):
        circle_delta = torch.zeros(
            (self.batch_size, self.circles.n, 2), 
            device=r2d.DEVICE)
        circle_delta[:, 0, :] = self.A_ACT_MUL * action
        return circle_delta

    def step(self, action, text=None):
        circle_delta = self.get_action(action)

        # state before update
        s = self.circles.state

        # update circle x and v
        for _ in range(self.frame_skip + 1):
            info = self.circles.update(circle_delta, self.dt, a_f=self.friction_coeff)
        s_next = self.circles.state
        reward = self.get_reward(s, action, s_next)
        self.t += 1

        # build info dict
        if text is None:
            text = ""
        text += f" T:{self.t}"
        info_dict = {} if not self.eval_mode else {
            "img": r2d.render(
                [self.circles], text=text,
                batch_idxs=range(self.batch_size), 
                canvas_size=2*self.X_OBS_MUL)
        }

        return (
            self.get_obs(),
            reward,
            False,
            info_dict
        )


    """ rendering """

    def _render_rgb(self, mode="rgb_array"):
        return r2d.render([self.circles], canvas_size=2*self.X_OBS_MUL)

    @torch.no_grad()
    def _render_traj_rgb(self, states, to_plot="object", **kwargs):

        traj_states = kwargs["traj"]["state"]
        # import code
        # code.interact(local=locals())
        
        init_state = self.circles.state
        b = self.batch_size
        T = states.shape[0]
        colors = self.get_colors()

        selector = np.arange(colors.shape[0]).tolist()
        if to_plot == "object":
            selector.remove(0)
        elif to_plot == "actor":
            selector.remove(1)

        def obj_iterator():
            """ avoid making a super large tensor that can blow up gpu memory """
            for t in range(T - 1):
                traj_states = kwargs["traj"]["state"][t]

                circles_state = traj_states[:, selector, :]
                circles_color = colors[selector, :]

                circles = r2d.Circles(circles_state, circles_color)
                yield circles
        
        text = f"epoch:{kwargs['info']['epoch_id']}"

        return r2d.render(
            obj_iterator(), 
            canvas_size=2*self.X_OBS_MUL, 
            text=text, 
            batch_idxs=np.arange(self.n_batches)
        )
