from tools.utils import animate, totensor
from solver.envs.softbody import CutEnv, RopeEnv

#env = CutEnv()
env = CutEnv(task='cut')

images =[]
for i in range(40):
    action = env.action_space.sample()
    action[1] = -1.
    action[0] = -1.
    action[2] = -1. if i > 20 else 0.
    env.step(totensor(action[None,:], device='cuda:0'))
    images.append(env.render(mode='rgb_array'))

animate(images, fps=10)