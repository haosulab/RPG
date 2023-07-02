import code
import torch
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def load_traj(path):
    return torch.load(path)

def entropy(classes: np.ndarray):
    p = np.histogram(classes, bins=np.unique(classes).shape[0])[0]
    p = p / p.sum()
    return -(p * np.log(p)).sum()

def classify(traj_batch):
    """ traj_batch: (b, t, d) """
    return ((traj_batch[:, :, 1] > 0).sum(-1) > (traj_batch.shape[1] // 2)).astype(int)

def visualize_traj(traj_batch, classes):
    points = traj_batch.reshape(-1, 2)
    plt.scatter(points[:, 0], points[:, 1], c=np.repeat(classes, traj_batch.shape[1]))

def build_traj_batch(traj):
    return traj['state'][:, :, 0, :2].transpose(0, 1).cpu().numpy()

if __name__ == "__main__":
    traj = load_traj("/home/litian/Desktop/RPG/Concept/solver/envs/rigidbody2d/exp/move1/debug/trajs/2")
    code.interact(local=locals())
