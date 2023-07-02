import code
from distutils.command.config import config
import sys
from tabnanny import check
from time import time
import cv2
import os
import torch
from tools.utils import animate
import tqdm
from solver.envs.rigidbody3d.utils import arr_to_str, write_text
from solver.envs.rigidbody3d.r3d_pick3cube import Pick3Cube
import nimblephysics as nimble
import matplotlib.pyplot as plt
from tools.utils import logger
from env import TestPullUp


def forward(i, env, a, render=False):
    env.reset()
    gamma = 1
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

        if render:
            text = f"iter:{i}-{t} r:{r.item():.5f}\n" \
                f"s_arm:{arr_to_str(env.sim.state[:7])}\n" \
                f"s_box:{arr_to_str(env.sim.state[7:13])}\n" \
                f"a    :{arr_to_str(a_t)}\n\n\n" \
                f"goals:{arr_to_str(env.goals)}\n" \
                f"ee   :{arr_to_str(env.end_effector_pos(env.sim.state))} \n" \
                f"box  :{arr_to_str(env.box_pos(env.sim.state, 0))}\n"
            img = env.render() # text=text
            imgs.append(img)
        rewards.append(r.squeeze().item())
    return dict(actions=actions, rewards=rewards, imgs=imgs, R=R.squeeze())


def plot_eps_return(eps_returns):
    plt.clf()
    plt.figure()
    plt.title("Episode Discounted Return"); plt.xlabel("episode"); plt.ylabel("discounted return")
    plt.plot(eps_returns); plt.grid()
    plt.savefig(f"{logger.get_dir()}/eps_discounted_return.png"); plt.close()


def detect_anomaly(i, env, R_prev, R, a_prev, a):

    if i % 100 == 0: # abs(R - R_prev) > 2 or 

        folder_name = f"{i:03d}"
        os.makedirs(f"{logger.get_dir()}/{folder_name}", exist_ok=True)

        a_prev = a_prev.detach().clone().requires_grad_(True)
        a = a.detach().clone().requires_grad_(True)

        logger.torch_save(a_prev.detach().clone(), f"{folder_name}/a_prev.pth")
        logger.torch_save(a.detach().clone(), f"{folder_name}/a.pth")

        traj_prev = forward(i - 1, env, a_prev, render=True)
        traj_prev["R"].backward()

        traj = forward(i, env, a, render=True)
        traj["R"].backward()

        stdout = sys.stdout
        with open("/dev/null", 'w') as f:
            sys.stdout = f
            logger.animate(traj_prev['imgs'], f"{folder_name}/{i-1:03d}.mp4", fps=20)
            logger.animate(traj['imgs'], f"{folder_name}/{i:03d}.mp4", fps=20)
        sys.stdout = stdout

        a_prev_grad_norm = a_prev.grad.norm(dim=-1)
        a_grad_norm = a.grad.norm(dim=-1)

        plt.clf(); plt.figure(); plt.title("grad norm"); plt.xlabel("timestep"); plt.ylabel("L2 norm")
        plt.plot(a_prev_grad_norm.numpy(), color="C0", label=f"{i-1:03d}")
        plt.plot(a_grad_norm.numpy(), color="C1", label=f"{i:03d}")
        plt.grid(); plt.legend(); plt.savefig(f"{logger.get_dir()}/{folder_name}/{i-1:03d}_{i:03d}_grad_norm.png")

        plt.clf(); plt.figure(); plt.title("traj reward"); plt.xlabel("timestep"); plt.ylabel("reward(t)")
        plt.plot(traj_prev["rewards"], color="C0", label=f"{i-1:03d}")
        plt.plot(traj["rewards"], color="C1", label=f"{i:03d}")
        plt.grid(); plt.legend(); plt.savefig(f"{logger.get_dir()}/{folder_name}/{i-1:03d}_{i:03d}_traj_reward.png")

def main():
    logger.configure("traj", ["stdout", "csv"])
    print("logdir setup at:", logger.get_dir())

    env = Pick3Cube()
    env.reset()
    print(env.A_ACT_MUL)

    # save actions before optimize
    a_prev = torch.ones((env.low_steps, env.action_space.shape[0])) * 0.001
    a = torch.nn.Parameter(a_prev.clone())

    traj_prev = None
    traj = None

    opt = torch.optim.Adam([a], lr=1e-2)

    # i = 0
    eps_returns = []
    bestR = float("-inf"); besti = 0
    # while True:
    for i in tqdm.trange(100000):
        # print(f">", end="", flush=True)

        if i % 100 == 0:
            print()
            traj = forward(i, env, a, render=True)
            os.makedirs(f"{logger.get_dir()}/render_cycle", exist_ok=True)
            logger.animate(traj["imgs"], f"render_cycle/{i:03d}.mp4", fps=20)
            os.system(f"cp {logger.get_dir()}/render_cycle/{i:03d}.mp4 {logger.get_dir()}/last.mp4")
            plot_eps_return(eps_returns)
        else:
            traj = forward(i, env, a, render=False)

        # if len(eps_returns) > 0:
        #     detect_anomaly(i, env, eps_returns[-1], traj["R"].item(), a_prev, a)

        if traj["R"] > bestR:
            best_action = a.detach().clone()
            bestR, besti = traj["R"], i

        if i % 100 == 0:
            with torch.no_grad():
                best_traj = forward(besti, env, best_action, render=True)
                logger.animate(best_traj["imgs"], f"best.mp4", fps=20)

        opt.zero_grad()
        (-traj["R"]).backward()
        eps_returns.append(traj["R"].item())

        print("action")
        print(arr_to_str(a[0]))

        print("reward")
        print(arr_to_str(traj["rewards"]))
        
        print("action grad")
        print(arr_to_str(a.grad.norm(dim=-1)))

        a_prev = a.detach().clone()
        opt.step()
        a.data[:] = torch.clamp(a.data, -5, 5)
        # i += 1


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    main()