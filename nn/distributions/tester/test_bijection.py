# NOTE: no need for bijection
import sys
import numpy as np
from torch.distributions import Normal
import torch


dim = 1
device = 'cuda:0'

mu = torch.nn.Parameter(torch.tensor(np.array([0.1]*dim), device=device))
log_std = torch.nn.Parameter(torch.tensor(np.array([0.3]*dim), device=device))
params = [mu, log_std]


import tqdm



# measure the convergence ..
mid0 = torch.tensor([0.5, 0.3, 0.2, 0.2], device='cuda:0')
mid1 = torch.tensor([-0.4, -0.3, 0.2, 0.2], device='cuda:0')

for i in tqdm.trange(10000):
    dist = Normal(mu, log_std.exp())
    s = dist.rsample((100000,))

    #s = s[:2] + s[:, 2:]
    ss = s

    s = abs(s)
    R1 = torch.exp(-torch.linalg.norm(s - mid0[:dim], axis=-1)**2 * 4)
    R2 = torch.exp(-torch.linalg.norm(s - mid1[:dim], axis=-1)**2 * 4) + 0.2

    R = torch.maximum(R1, R2)
    #flag = (s[:, 0] < 0.5).float()
    #R = R1 * flag + R2 * (1-flag)

    if sys.argv[1] == "GD":
        (-R).mean().backward()
    else:
        logp = torch.log(torch.exp(dist.log_prob(s.detach()).sum(axis=-1)) + torch.exp(dist.log_prob(-s.detach()).sum(axis=-1)))
        (-logp * R.detach()).mean().backward()
    
print(mu.grad[:])
print(log_std.grad[:])