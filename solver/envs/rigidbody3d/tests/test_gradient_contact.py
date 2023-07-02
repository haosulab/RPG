import torch
from solver.envs.rigidbody3d.test_aseq_grasp import PickUp
from solver.envs.rigidbody3d.utils import arr_to_str


env = PickUp()
env.reset()

a = torch.load("tests/test_gradient_contact/a_iter200.tensor")


a = a.detach().clone()

stop = 40
action_47 = a[stop].clone()
action_47[0:3] = 0
action_47[3] = 5 * 0.0002
action_47[4] = 0
action_47[5:7] *= 1
a[stop:] = action_47


a = torch.nn.Parameter(a)
opt = torch.optim.Adam([a], lr=0)

env.sim.viewer.create_window()

for i in range(1, env.low_steps):
    env.reset()
    opt.zero_grad()
    R = 0

# i = env.low_steps

    for j in range(i):
        s_, r, done, info = env.step(a[j])
        # R += r
        env.render()

    env.box_pos(env.sim.state, 0)[1].backward()
    if i > 1:
        # print(i, arr_to_str(a.grad[:i][-2] * 1e8, 9))
        print(i, a.grad[:i][-2])
    opt.step()


while not env.sim.viewer.window.closed:
    env.render()

