import gym
import numpy as np
import matplotlib.pyplot as plt
from tools import Configurable, as_builder


@as_builder
class GoalEnv(Configurable, gym.Env):
    """
    Goal Env has the following properties:
    - support multi goals and set, get
    - the get and set is all in the batch mode

    - each step returns a batch of obs, reward, done (always False), infos
        - observations and rewards should be differetiable
        - next obs could be None for single stage tasks ..
    """

    def __init__(self, cfg=None, num_stages=1, low_steps=10):
        gym.Env.__init__(self)
        Configurable.__init__(self)

        self.num_stages = num_stages
        self.low_steps = low_steps

    def sample_state_goal(self, batch_size=1):
        raise NotImplementedError

    def get_state_goal(self):
        return self.get_state(), self.goal

    def reset_state_goal(self, states, goals):
        # always requires grad ..
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

    def reset(self, states=None, goals=None, batch_size=1, return_state_goal=False):
        self._nsteps = 0
        if states is None:
            states, goals = self.sample_state_goal(batch_size=batch_size)

        if return_state_goal:
            return states, goals

        self.reset_state_goal(states, goals)
        return self.get_obs()

    def get_obs(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError
    
    def _render_rgb(self, index=0):
        raise NotImplementedError

    def _render_traj_rgb(self, **kwargs):
        # used to render the trajectories/ a set of states together .. useful to visualize the trajectories. 
        raise NotImplementedError("traj renderer does not implemented yet.. ")

    @property
    def batch_size(self):
        raise NotImplementedError

    def render(self, mode='rgb_array', num_render=1, **kwargs):
        images = []
        for index in range(min(num_render, self.batch_size)):
            images.append(self._render_rgb(index, **kwargs))
        images = np.concatenate(images, 1)
        if mode == 'plt':
            plt.imshow(images)
            plt.show()
            return
        elif mode.endswith('.png'):
            plt.imshow(images)
            plt.savefig(mode)
            return
        return images