# move to three locations
from tools.utils import totensor
import numpy as np
import cv2
from solver.envs.point_env import PointEnv2

class TripleMove(PointEnv2):
    def __init__(self, cfg=None, n_goals=3, reward_weight=0.1, action_penalty=0., goal_type=0):
        super().__init__()

        if goal_type == 0:
            self.goals = totensor(
                [
                    [0                      , -1.], 
                    [1./2 * 3 ** 0.5 , 1./2],
                    [-1./2 * 3 ** 0.5, 1./2]
                ], 
                device='cuda:0'
            )[:n_goals] * 0.4 + 0.5
        else:
            self.goals = totensor(
                [
                    [0                      , 0.8], 
                    [1./2 * 3 ** 0.5 , 1./2],
                    [-1./2 * 3 ** 0.5, 1./2]
                ], 
                device='cuda:0'
            )[:n_goals] * 0.4 + 0.5
        print(self.goals)

    def sample_state_goal(self, batch_size=1):
        state = np.zeros((batch_size, 3))
        state[:, :2] = 0.5
        return state, self.goals

    def get_obstacles(self):
        self.obstacle_penalty = 0.

    def get_reward(self):
        return -(self.state[:, None, :2] - self.goals[None, :, :2]).norm(dim=-1).min(dim=-1)[0] * self._cfg.reward_weight

    def obstalce_impulse(self):
        return

    def _render_rgb(self, index=0):
        # only render the first ..
        size = 128
        start = size//2
        x, y = self.state[index][:2].detach().cpu().numpy()
        img = np.zeros((size*2, size*2, 3), dtype=np.uint8) + 64

        def f(x):
            return int(x * size) + start
        def ff(x, y):
            return (f(x), f(y))

        img = cv2.circle(img, ff(x, y), 3, (255, 0, 0), -1)
        for i in self.goals:
            img = cv2.circle(img, ff(*i), int(0.1 * size), (0, 255, 0), -1)

        if self._cfg.boundary:
            img = cv2.rectangle(img, (start, start), (size+start, size+start), (255, 0, 255), 1, 1)
        return img

