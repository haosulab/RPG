import numpy as np
from tools.utils.rrt import RRTConnectPlanner


def state_sampler():
    return np.random.uniform(0, 1, 2)

def collision_checker(state):
    for cx, cy in zip([0.25, 0.25, 0.75, 0.75], [0.25, 0.75, 0.25, 0.75]):
        dx = state[0] - cx
        dy = state[1] - cy
        d = ((dx ** 2) + (dy ** 2)+1e-9)**0.5
        #gap = 0.15 - d
        if d < 0.15:
            return True
    return False

def get_tree():
    rrt = RRTConnectPlanner(state_sampler, collision_checker, expand_dis=0.1, step_size=0.01, max_iter=10000)

    path = rrt(np.array([0., 0.]), np.array([1., 1]), info=True, return_tree=True, rrt_connect=False)
    return path
