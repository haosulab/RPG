import torch
import numpy as np
import matplotlib.pyplot as plt
import glob

from tools.config import CN

device = "cuda"
rpg_path = f"data/bandit_data/bandit_data/rpg/env_rdscale/0.5/banditx/1234/trajs/*"
reinforce_path = "data/bandit_data/bandit_data/pg/ent_a/0.1/banditx/1234/trajs/*"
linewidth = 2.0

def select(q):
    return sorted(list(glob.iglob(q)), key=lambda x: int(x.split("/")[-1]))
    # return sorted(os.listdir(q), key=lambda x: int(x.split("/")[-1]))
def load_epochs(q):
    epochs = []
    for path in select(q):
        # print(path)
        try:
            epoch = torch.load(path)
        except:
            pass
        epochs.append(epoch)
    return epochs



bandit1 = CN(dict(
    name="Bandit1",
    rdscale=0.5, discontinuity=True, 
    lcenter=-1.2, rcenter=1.75, lvshift=-0.6, rvshift=0.0,
    ldscale=6.0, lescale=1.0, rescale=1.0))

bandit2 = CN(dict(
    name="Bandit2",
    rdscale=1, discontinuity=False, 
    lcenter=-1.2, rcenter=1.75, lvshift=-0.6, rvshift=0.0,
    ldscale=6.0, lescale=1.0, rescale=1.0))

def reward(x, cfg):
    dscale = torch.tensor([cfg.ldscale, cfg.rdscale], device=device).reshape(-1, 1)
    eshift = torch.tensor([cfg.lvshift, cfg.rvshift], device=device).reshape(-1, 1)
    escale = torch.tensor([cfg.lescale, cfg.rescale], device=device).reshape(-1, 1)
    intersect_left = cfg.rcenter - np.sqrt(1-2*cfg.rvshift)/cfg.rdscale
    # intersect_right = cfg.rcenter + np.sqrt(1-2*cfg.rvshift)/cfg.rdscale
    rcenter = cfg.rcenter - (intersect_left + 0.25)
    action = x
    c = torch.tensor([cfg.lcenter, rcenter], device=device).unsqueeze(1)
    x = x.squeeze().unsqueeze(0).repeat(c.shape[0], 1)
    d = (x - c) * dscale
    e = (0.5 * escale * d ** 2) + eshift
    e = e.min(dim=0)[0]
    r = torch.clamp(-e, min=-0.5)
    if cfg.discontinuity:
        jump = c[1]
        return r * (action.squeeze() < jump).float() + (action.squeeze() > jump).float() * (-0.5)
    else:
        return r

def plot_reward(cfg, color):
    x = torch.linspace(-3, 3, 200).unsqueeze(1).to(device)
    y = reward(x, cfg) + 0.5
    plt.xlim(-2.5, 2.5); plt.ylim(-0.1, 1.2)
    plt.xlabel("action"); plt.ylabel("reward")
    plt.plot(x.reshape(-1).cpu().numpy(), y.reshape(-1).cpu().numpy(), linewidth=linewidth, color=color)

def dist_hist(actions, bins):
    x_freq = np.linspace(-2.5, 2.5, bins + 1)
    inds = np.digitize(actions, x_freq)
    freq = np.zeros_like(x_freq)
    unique, counts = np.unique(inds, return_counts=True)
    for i, count in zip(unique, counts):
        freq[i-1] = count
    freq = freq / freq.sum() * 2
    return x_freq + (x_freq[1] - x_freq[0]) / 2, freq/0.6


def plot_action_distribution(epoch, bins=50, label=None, color=None):
    a = torch.stack([t["a"] for t in epoch]).reshape(-1).cpu().numpy()
    z = torch.stack([t["z"] for t in epoch]).reshape(-1).cpu().numpy()
    #print(z)
    #exit(0)
    x_freq, freq = dist_hist(a, bins)
    gap = x_freq[1] - x_freq[0]
    x_freq = np.append(x_freq - gap /2, x_freq[-1] + gap / 2)
    freq *= 0.5
    plt.stairs(freq, x_freq, color='#0504aa', fill=True)
    # plt.fill_between(x_freq, np.zeros_like(freq), freq, alpha=0.5, label=label, color=color)
    plt.scatter(a, np.zeros_like(a) - 0.05, c=z, cmap="jet", s=10)


if __name__ == '__main__':
    from tools.utils import animate, plt_save_fig_array

    data = dict(
        Reinforce=load_epochs(reinforce_path),
        RPG=load_epochs(rpg_path)
    )

    reward_cfg = bandit1

    import tqdm
    epochs_to_plot = 80
    video_frames = []
    fig_width = 5

    import os
    os.makedirs("data/plots/bandit", exist_ok=True)
    for i in tqdm.trange(epochs_to_plot):
        for j, method in enumerate(data):
            plt.figure(figsize=(fig_width, fig_width))
            #plt.subplot(1, len(data), j + 1)
            # plt.title(f"{method}")
            plot_reward(reward_cfg, color="green")
            plot_action_distribution(data[method][i], bins=20)
            plt.tight_layout()
        #video_frames.append(plt_save_fig_array())
            plt.savefig("data/plots/bandit/" + f"{method}_{i:03d}.png")
    #animate(video_frames, f"{reward_cfg.name}.mp4", fps=10)

