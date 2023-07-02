# used to visualize the envs with single line of code ..
import numpy as np
import argparse
import cv2
from solver import *
from solver.envs.goal_env_base import GoalEnv
import matplotlib.pyplot as plt
from tools.utils import animate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name')
    args = parser.parse_args()

    env: GoalEnv = GoalEnv.build(TYPE=args.env_name)
    o = []
    o.append(env.reset())

    a = env.action_space.sample()
    print('aciton shape', a.shape)
    images = []
    states = []
    for i in range(50):
        o.append(env.step([env.action_space.sample()])[0])

        rgb = env._render_rgb()
        images.append(rgb)

        states.append(env.get_state())

    animate(images)

    import torch
    image2 = env._render_traj_rgb(torch.stack(o))
    cv2.imwrite('trajs.png', image2[...,::-1])




if __name__ == '__main__':
    main()