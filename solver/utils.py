import torch
from gym.spaces import Box, Discrete
from collections import defaultdict
import numpy as np
from tools.utils import batch_input

from nn.space import MixtureSpace
from nn.distributions import NormalAction, MixtureAction, CategoricalAction
from tools.utils import myround



def create_z_space(z_dim, z_cont_dim):
    if z_cont_dim == 0:
        z_space = Discrete(z_dim)
    elif z_dim == 0:
        z_space = Box(-1, 1, (z_cont_dim,))
    else:
        z_space = MixtureSpace(Discrete(z_dim), Box(-1, 1, (z_cont_dim,)))
    return z_space


def group_videos(images, target_size=(128*3, 128)):
    import cv2
    for i in images:
        for j in range(len(i)):
            i[j] = cv2.resize(i[j], target_size)
    outs = []
    for i in range(len(images[0])):
        outs.append(np.concatenate([j[i] for j in images], 0))
    return outs


def create_normal_prior(dim, std=1., device='cuda:0'):
    #device = self.pi_a.device
    zero = batch_input(np.zeros(dim), device=device)
    return NormalAction(zero, zero + std)


def create_prior(space, device='cuda:0'):
    if isinstance(space, Box):
        assert isinstance(space, Box)
        return create_normal_prior(space.shape, std=1., device=device)
    elif isinstance(space, Discrete):
        #z_space = self.pi_z.action_space
        assert isinstance(space, Discrete)
        return CategoricalAction(logits=torch.zeros((space.n,), device=device))
    else:
        discrete = create_prior(space.discrete, device=device)
        continuous = create_prior(Box(-1, 1, (space.discrete.n, space.continuous.shape[0])), device=device)
        return MixtureAction(discrete, continuous)


def reverse_cumsum(rewards, dim=0):
    return rewards + rewards.sum(dim, keepdims=True) - torch.cumsum(rewards, dim=dim)