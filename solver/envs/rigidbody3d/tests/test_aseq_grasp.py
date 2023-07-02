import code
import cv2
import os
import torch
from .r3d_grasp import GraspBox
from tools.utils import animate
import tqdm
from utils import arr_to_str, write_text
import nimblephysics as nimble
import matplotlib.pyplot as plt


class Reach(GraspBox):

    def __init__(self, cfg=None):
        super().__init__(cfg)
        # self.sim.enforce_limit = enforce_limit
    
    def get_reward(self, s, a, s_next):
        # try to reach rhs first, nothing too fancy
        ee_pos_s  = nimble.map_to_pos(self.sim.world, self.sim.arm_fk, s).float().reshape(-1, 3)
        dist_before = (ee_pos_s - self.goals).norm(-1).sum()
        # assert ee_pos_s.shape == torch.Size([3, 3])
        ee_pos_s_next = nimble.map_to_pos(self.sim.world, self.sim.arm_fk, s_next).float().reshape(-1, 3)
        dist_after = (ee_pos_s_next - self.goals).norm(-1).sum()
        reach = (dist_before - dist_after).norm()
        # assert ee_pos_s_next.shape == torch.Size([3, 3])

        # help enforce gripper pos joint limit
        gripper_enforce = - (s_next[5] ** 2 + s_next[6] ** 2)
        # gripper_enforce = 0

        # velocity penalty
        # velocity_penalty = ((ee_pos_s_next - ee_pos_s).norm(-1) ** 2).sum() / 200
        return (reach + gripper_enforce) # + velocity_penalty

class PickUp(GraspBox):

    def __init__(self, cfg=None):
        super().__init__(cfg)
        # self.sim.enforce_limit = enforce_limit
        self.sim.gripper_pad_fk = nimble.neural.IKMapping(self.sim.world)
        for bodyNode in ["left_pad", "right_pad"]:
            self.sim.gripper_pad_fk.addLinearBodyNode(
                self.sim.arm.nimble_actor.getBodyNode(bodyNode))

    def gripper_center(self, state):
        return nimble.map_to_pos(self.sim.world, self.sim.gripper_pad_fk, state).float().view(2, 3).mean(0)

    def closest_box(self, s):
        # with torch.no_grad():
        #     ee_pos_s  = nimble.map_to_pos(self.sim.world, self.sim.arm_fk, s).float()
        #     box_index = torch.argmin(torch.tensor([(self.box_pos(s, i) - ee_pos_s).norm() for i in range(len(self.sim.box))])).item()
        # return box_index
        return 0

    @torch.no_grad()
    def gripper_is_on_top(self, s):
        box_index_s = self.closest_box(s)
        box_pos_s = self.box_pos(s, box_index_s)
        ee_pos_s = nimble.map_to_pos(self.sim.world, self.sim.arm_fk, s).float()
        disp_before = (ee_pos_s - box_pos_s)
        return (disp_before[[0, 2]].norm() < 0.2) and (disp_before[1].abs() < 0.55)

    @torch.no_grad()
    def time_to_pull_up(self, s):
        ee_vshift = torch.tensor([0, -0.35, 0])
        # select box and reach
        box_index_s = self.closest_box(s)
        # reach ee to box
        box_pos_s = self.box_pos(s, box_index_s)
        # box_pos_s = torch.tensor([0, 0.7, -1.5], dtype=torch.float32)
        ee_pos_s = nimble.map_to_pos(self.sim.world, self.sim.arm_fk, s) + ee_vshift
        # ee_pos_s = self.gripper_center(s)
        disp_before = (ee_pos_s - box_pos_s)
        dist_before = disp_before.norm()
        return s[5] > 0.09 and s[6] > 0.09 and dist_before < 0.2

    def get_action(self, action):
        return super().get_action(action)

    def get_reward(self, s, a, s_next):
        """ working version fric_coeff=1. restitution=0.1 lr=1e-2"""
        # try pull up first
        self.goals = torch.tensor([0, 1, 1.5])

        self.text = "\n"
        ee_vshift = torch.tensor([0, -0.4, 0])
        # select box and reach
        box_index_s = self.closest_box(s)

        # reach ee to box
        box_pos_s = self.box_pos(s, box_index_s)
        ee_pos_s = nimble.map_to_pos(self.sim.world, self.sim.arm_fk, s) + ee_vshift
        disp_before = (ee_pos_s - box_pos_s)
        dist_before = disp_before.norm()

        box_pos_s_next = self.box_pos(s_next, box_index_s)
        ee_pos_s_next = nimble.map_to_pos(self.sim.world, self.sim.arm_fk, s_next) + ee_vshift

        disp_after = (ee_pos_s_next - box_pos_s_next)
        dist_after = disp_after.norm()

        reach = (dist_before - dist_after)
        self.text += f"reward reach:{reach:.4f}\n"
        
        # box_pos_diff = (box_pos_s_next - box_pos_s)
        # pull_up = (box_pos_diff[1] - box_pos_diff[[0, 2]].norm()) * 10
        pad_to_box_s      = nimble.map_to_pos(self.sim.world, self.sim.gripper_pad_fk, s).float().view(2, 3) - box_pos_s
        pad_to_box_s_next = nimble.map_to_pos(self.sim.world, self.sim.gripper_pad_fk, s_next).float().view(2, 3) - box_pos_s_next
        # pull_up = -(s_next[3] - s[3]) + box_pos_diff[1]

        if self.t > 30: # close gripper
            gripper_enforce = -((s_next[5] - 0.11) ** 2 + (s_next[6] - 0.11) ** 2)
            # gripper_enforce = (pad_to_box_s.norm() - pad_to_box_s_next.norm())
            gripper_enforce += -(self.gripper_center(s_next) - ee_pos_s_next)[[0,2]].norm() ** 2
            self.text += f"reward gripper_enforce:{gripper_enforce:.4f}\n"

            box_goal_diff_s      = (box_pos_s - self.goals)
            box_goal_diff_s_next = (box_pos_s_next - self.goals)
            dist_box_goal_before = box_goal_diff_s.norm()
            self.text += f"reward dist_box_goal:{dist_box_goal_before}\n"
            dist_box_goal_after  = box_goal_diff_s_next.norm()
            box_to_goal = (dist_box_goal_before - dist_box_goal_after)
            self.text += f"reward box_to_goal:{box_to_goal:.4f}\n"
            
            return (reach + gripper_enforce + box_to_goal)
        else:
            gripper_enforce = -(s_next[5] ** 2 + s_next[6] ** 2)
            self.text += f"reward gripper_enforce:{gripper_enforce:.4f}\n"
            return (reach + gripper_enforce)


class PickUp(GraspBox):

    def __init__(self, cfg=None, 
        low_steps=100, dt=0.005, frame_skip=0, 
        restitution_coeff=0.0, friction_coeff=1., 
        n_batches=1, X_OBS_MUL=5., V_OBS_MUL=5., A_ACT_MUL=0.002
    ):
        super().__init__(cfg, low_steps, dt, frame_skip, restitution_coeff, friction_coeff, n_batches, X_OBS_MUL, V_OBS_MUL, A_ACT_MUL)

    def get_reward(self, s, a, s_next):
        """ working version eps: 3700"""
        self.goals = torch.tensor([0, 1, 1.5])
        ee  = nimble.map_to_pos(self.sim.world, self.sim.arm_fk, s) + torch.tensor([0, -0.45, 0])
        box = self.box_pos(s_next, 0)
        reach = -(ee - box).norm() ** 2
        if self.t < 30:
            gripper_enforce = -(s_next[5] ** 2 + s_next[6] ** 2)
            return (reach + gripper_enforce)
        else:
            box_to_goal = -(box - self.goals).norm() ** 2
            gripper_enforce = -((s_next[5] - 0.11) ** 2 + (s_next[6] - 0.11) ** 2)
            pull_up = -s_next[3]
            return (0.2 * reach + gripper_enforce + 1.5 * box_to_goal + pull_up * 0.1)

class PickUp(GraspBox):

    def __init__(self, cfg=None, 
        low_steps=100, dt=0.005, frame_skip=0, 
        restitution_coeff=0.2, friction_coeff=1., 
        n_batches=1, X_OBS_MUL=5., V_OBS_MUL=5., A_ACT_MUL=0.002
    ):
        super().__init__(cfg, low_steps, dt, frame_skip, restitution_coeff, friction_coeff, n_batches, X_OBS_MUL, V_OBS_MUL, A_ACT_MUL)

    def get_reward(self, s, a, s_next):
        """ ball pick working version """
        self.goals = torch.tensor([0, 0.5, 1.5])
        ee  = nimble.map_to_pos(self.sim.world, self.sim.arm_fk, s) + torch.tensor([0, -0.2, 0])
        box = self.box_pos(s_next, 0)
        reach = -(ee - box).norm() ** 2
        if self.t < 30:
            gripper_enforce = -(s_next[5] ** 2 + s_next[6] ** 2)
            return (reach + gripper_enforce)
        elif self.t < 40:
            gripper_enforce = -((s_next[5] - 0.11) ** 2 + (s_next[6] - 0.11) ** 2)
            return reach + gripper_enforce
        else:
            gripper_enforce = -((s_next[5] - 0.11) ** 2 + (s_next[6] - 0.11) ** 2)
            box_to_goal = -(box - self.goals).norm() ** 2
            return gripper_enforce + (box_to_goal)


class PickUp(GraspBox):

    def __init__(self, cfg=None, 
        low_steps=100, dt=0.005, frame_skip=0, 
        restitution_coeff=0.2, friction_coeff=1., 
        n_batches=1, X_OBS_MUL=5., V_OBS_MUL=5., A_ACT_MUL=0.002
    ):
        super().__init__(cfg, low_steps, dt, frame_skip, restitution_coeff, friction_coeff, n_batches, X_OBS_MUL, V_OBS_MUL, A_ACT_MUL)

    def get_reward(self, s, a, s_next):
        self.goals = torch.tensor([0, 0.5, 1.5])
        ee  = nimble.map_to_pos(self.sim.world, self.sim.arm_fk, s) + torch.tensor([0, -0.2, 0])
        box = self.box_pos(s_next, 0)
        reach = -(ee - box).norm() ** 2
        if self.t < 20:
            gripper_enforce = -(s_next[5] ** 2 + s_next[6] ** 2)
            return (reach + gripper_enforce)
        elif self.t < 40:
            gripper_enforce = -((s_next[5] - 0.11) ** 2 + (s_next[6] - 0.11) ** 2)
            return reach + gripper_enforce
        else:
            gripper_enforce = -((s_next[5] - 0.11) ** 2 + (s_next[6] - 0.11) ** 2)
            box_to_goal = -(box - self.goals).norm() ** 2
            return gripper_enforce + (box_to_goal)

if __name__ == "__main__":

    dir_name = "tests/test_aseq_grasp"
    video_name = "reach.mp4"
    os.system(f"rm {dir_name}/[0-9]*.mp4")
    os.system(f"rm {dir_name}/[0-9]*.txt")
    os.system(f"mkdir  {dir_name}")

    env = PickUp()
    env.reset()
    # cv2.imwrite("debug.png", env.render())
    # input()

    gamma = 1
    a = torch.nn.Parameter(torch.ones((env.low_steps, env.action_space.shape[0])) * 0.001)
    # print(a.shape)
    opt = torch.optim.Adam([a], lr=1e-2)
    
    i = 0
    eps_returns = []
    # env.sim.viewer.create_window()
    # while not env.sim.viewer.window.closed:
    render_cycle = 100
    while True:
        print(f">", end="", flush=True)

        actions = []
        rewards = []
        imgs    = []
        R = 0

        s = env.reset()

        for t in range(env.low_steps):
            a_t = a[t]
            actions.append(a_t)

            state = env.sim.state
            s_, r, done, info = env.step(a_t)

            R += gamma ** t * r
            s = s_

            # f"dr/da:{j_ra}\n"\
            text = f"iter:{i}-{t} r:{r.item():.5f}\n" \
                f"s_arm:{arr_to_str(env.sim.state[:7])}\n" \
                f"s_box:{arr_to_str(env.sim.state[7:13])}\n" \
                f"a    :{arr_to_str(a_t)}\n\n\n" \
                f"goals:{arr_to_str(env.goals)}\n" \
                f"ee   :{arr_to_str(env.end_effector_pos(env.sim.state))} \n" \
                f"box  :{arr_to_str(env.box_pos(env.sim.state, 0))}\n"
                # f"gripc:{arr_to_str(env.gripper_center(env.sim.state))}\n" \
                # f"gripper_on_top:{env.gripper_is_on_top(env.sim.state)}\n"
                # f"time_to_pull:{env.time_to_pull_up(env.sim.state)}\n" \
            # text = None
            if (i % render_cycle == 0):
                img = env.render(text=text)
                imgs.append(img)
            rewards.append(r)
        
        opt.zero_grad()
        (-R.mean()).backward()
        eps_returns.append(R.item())

        if (i % render_cycle == 0):
            torch.save(a, f"{dir_name}/{i}.txt")
            print()
            plt.title("Episode Discounted Return"); plt.xlabel("episode"); plt.ylabel("discounted return")
            plt.plot(eps_returns); plt.grid()
            plt.savefig(f"{dir_name}/eps_discounted_return.png"); plt.close()

            print(f"\niter:{i}")
            # log action grad and stuff
            print("rewards")
            for r in rewards:
                print(f"{r.item():.3f}", end="|")
            print()

            print("action grad norm")
            print(len(imgs), len(a.grad))
            for t, a_grad in enumerate(a.grad):
                print(f"{a_grad.norm(dim=-1).item():.3f}", end="|")
                imgs[t] = write_text(imgs[t], f"\n\n\n\n\nagrad:{arr_to_str(a_grad)}")
            print()

            print(f"R={R.item()}")

            animate(imgs, f"{dir_name}/{i:03d}.mp4", fps=20)
            os.system(f"cp {dir_name}/{i:03d}.mp4 {dir_name}/{video_name}.mp4")
            print(f"{env.end_effector_pos(env.sim.state) = }", f"{env.goals = }")

        # while not env.sim.viewer.window.key_down_action("n"):
        #     env.sim.update_viewer()

        opt.step()
        i += 1
