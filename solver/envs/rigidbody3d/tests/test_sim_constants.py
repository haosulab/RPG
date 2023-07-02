import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import nimblephysics as nimble
from solver.envs.rigidbody3d.r3d_grasp import GraspBox
from solver.envs.rigidbody3d.utils import arr_to_str

class TestSim(GraspBox):

    def __init__(self, cfg=None):
        super().__init__(cfg, A_ACT_MUL=1.)

    def init_simulator(self):

        self.sim.world.setGravity([0, -1, 0])

        # arm
        self.sim.arm    = self.sim.load_urdf(
            "gripper_v2.urdf", 
            restitution_coeff=self.restitution_coeff,
            friction_coeff=self.friction_coeff,
            mass=1)
        
        # boxes
        self.sim.box = [
            self.sim.load_urdf(
                "sphere.urdf", 
                restitution_coeff=self.restitution_coeff,
                friction_coeff=self.friction_coeff,
                mass=0.1, inertia=[0.1, 0.1, 0.1])
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

    def sample_state_goal(self, batch_size=1):
        state, goals = super().sample_state_goal(batch_size)

        # prismatic qpos
        state[3] = state[3] + 0.4522
        # gripper qpos
        # state[]
        
        state[5:7] = torch.tensor([0.0, 0.0])
        ee = nimble.map_to_pos(self.sim.world, self.sim.arm_fk, state)
        state[10:13] = ee + torch.tensor([0, -0.45, 0])
        return state, goals

    def sample_state_goal(self, batch_size=1):
        # set up init pos & qpos & goal
        box_x = 0.4
        box_y = (1 - box_x ** 2) ** 0.5
        state = torch.tensor(
            # [   
            #     # arm qpos
            #     -1.57 / 2 - 0.2, 0.7, -0.7, 0.2822, 2.22, 0.1, 0.1,
            #     0, 0, 0,    0.058, -0.404, 1.652,
            #     # velocities
            #     0, 0, 0, 0, 0, 0, 0,
            #     0, 0, 0, 0, 0, 0,
            # ], dtype=torch.float32
            [   
                # arm qpos
                -1.57 / 2 - 0.2, 0.7, -0.7, 0.2822, 2.22, 0.0, 0.0,
                0, 0, 0,    0.058, -0.404, 1.652,
                # velocities
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
            ], dtype=torch.float32
        )
        goals = torch.tensor([0, 1, -1.5], dtype=torch.float32)
        return state, goals

    def get_reward(self, s, a, s_next):
        # r_mult = 0.1
        # ee  = nimble.map_to_pos(self.sim.world, self.sim.arm_fk, s_next) \
        #     + torch.tensor([0, -0.2, 0])
        # box = self.box_pos(s_next, 0)
        # reach = -(ee - box).norm() ** 2

        # gripper_qpos_at_0 = -(s_next[5] ** 2 + s_next[6] ** 2)
        # gripper_close = -((s_next[5] - 0.12) ** 2 + (s_next[6] - 0.12) ** 2)
        
        # if self.t < 20:
        #     return (reach + gripper_qpos_at_0) #* r_mult
        # elif self.t < 40:
        #     return (reach + gripper_close * 5) #* r_mult
        # else:
        #     box_to_goal = -(box - self.goals).norm() ** 2
        #     return gripper_close * 5 + (box_to_goal) * r_mult * 5
        return 0


def forward(env, a, T, render, render_cycle=10):
    env.reset()
    imgs = []
    slip_diffs = []

    s = env.sim.state
    for t in range(T):
        assist = env.sim.get_assist()
        if t >= 85:
            assist[-2:] = 0
        env.step(a[t] + assist)
        s_next = env.sim.state

        if render and t % render_cycle == 0:
            imgs.append(env.render())

        # when arm is pulling
        # if (s[3] - s_next[3]).abs() > 1e-5:
        # if a[t, 3] != 0:
        if t > 70:
            # how much did ball slipped
            slip_diff = (s[3] - s_next[3]).abs() - (s[11] - s_next[11]).abs()
            # slip_diff = -s[11]
            # print(t, slip_diff)
            slip_diffs.append(slip_diff.item())
        if t == 90:
            print((env.sim.get_assist())[3].item())
            
        s = s_next
    return locals()


def test(f_pull=-10, f_hold=0.1, T=100, rest=0.1, fric=1., render=True, gui=True):
    env = TestSim(restitution_coeff=rest, friction_coeff=fric)
    if gui: env.sim.viewer.create_window()

    a = torch.zeros((T, env.action_space.shape[0]))
    a[:85, -2:] = 50
    a[85:, 3] = f_pull; a[85:, -2:] = f_hold

    traj = forward(env, a, T, render=render)

    # print(traj['slip_diffs'])
    return traj


if __name__ == "__main__":
    import tqdm
    output_dir = "test_sim_constants"
    os.makedirs(output_dir, exist_ok=True)

    # N = 100
    # f_pull_interval = (-100, -10)
    # plt.figure(); plt.title(f"how much pulling force {f_pull_interval} influence slip")
    # plt.xlabel("simulator timestep"); plt.ylabel("delta height of (arm - box)")
    # for f_pull in np.linspace(*f_pull_interval, N):
    #     traj = test(f_pull=f_pull, render=False, gui=False)
    #     print(f_pull, traj['slip_diffs'][-1])
    #     plt.plot(traj["slip_diffs"])
    # plt.savefig(f"{output_dir}/pull_slip_{N}.png")


    # N = 20
    # f_pull_interval = (-100, -10)
    # plt.figure(); plt.title(f"how much pulling force {f_pull_interval} influence slip")
    # plt.xlabel("simulator timestep"); plt.ylabel("slip = delta height of (arm - box)")
    # for f_pull in np.linspace(*f_pull_interval, N):
    #     traj = test(f_pull=f_pull, render=False, gui=False)
    #     # print(f_pull, traj['slip_diffs'][-1])
    #     plt.plot(traj["slip_diffs"])
    # plt.savefig(f"{output_dir}/pull_slip_{N}.png")

    N = 20
    f_hold_interval = (0, 1000)
    plt.figure(); plt.title(f"how gripper pad force {f_hold_interval} influence slip")
    plt.xlabel("timestep"); plt.ylabel("delta height of (arm - box)")
    for i, f_hold in zip(tqdm.trange(N), np.linspace(*f_hold_interval, N)):
        traj = test(f_hold=f_hold, render=False, gui=False)
        # print(f_hold, traj['slip_diffs'][-1])
        plt.plot(traj["slip_diffs"])
    plt.savefig(f"{output_dir}/grip_slip_{N}.png")


    # test(f_hold=0, render=True, gui=True)