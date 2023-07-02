import os
os.makedirs('plots', exist_ok=True)

import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from solver.envs.softbody.plot_envs import TripleCont

fig = plt.figure(figsize=(10, 10))

batch_size = 200
env = TripleCont(action_scale = 0.1, low_steps=6)


nbins = 100
xmin, xmax = 0., 1.0
ymin, ymax = 0., 1.0
xbins = np.linspace(xmin, xmax, nbins)
ybins = np.linspace(ymin, ymax, nbins)


from tools.utils import totensor
pp = totensor(np.meshgrid(xbins, ybins), device='cuda:0').permute(1, 2, 0)
r =  -(pp[:, :, None, :2] - env.goals[None, None, :, :2]).norm(dim=-1).min(dim=-1)[0]

pp = pp.detach().cpu().numpy()
R =  ((r - r.min())/ (r.max() - r.min())).detach().cpu().numpy()

cp = plt.contourf(xbins, ybins, R)

# plt.colorbar(cp) # Add a colorbar to a plot


observations = [env.reset(batch_size=batch_size)]
reward = 0

for i in range(6):
    #action = env.action_space.sample()
    action = torch.rand(batch_size, 2).cuda() * 2 - 1
    obs, r = env.step(action)[:2]
    observations.append(obs)

    reward = reward + r 


def compute_density(states):
    states = states.reshape(-1, states.shape[-1])
    states = states[:, :2].detach().cpu().numpy() + np.random.normal(0, 0.03, (len(states), 2))
    #xmin, xmax = states[:, 0].min(), states[:, 0].max()
    #ymin, ymax = states[:, 1].min(), states[:, 1].max()
    density, _, _ = np.histogram2d(states[:, 0], states[:, 1], bins=[nbins, nbins])
    return density.transpose(1, 0)
#points = torch.cat(observations, axis=0).detach().cpu().numpy()
#density = compute_density(torch.cat(observations, axis=0))

#plt.imshow((density/min(density.max(), 100.)).clip(0., 1.), extent=[0., 1., 0., 1.])
traj = torch.stack(observations, axis=0).detach().cpu().numpy()
observations = torch.cat(observations, axis=0).detach().cpu().numpy()
plt.scatter(observations[:, 0], observations[:, 1], c='b', s=15)
plt.xlim(0., 1.)
plt.ylim(0., 1.)

plt.tight_layout()
plt.savefig('plots/prior.png')

plt.clf()

cp = plt.contourf(xbins, ybins, R)
#plt.colorbar(cp) # Add a colorbar to a plot

plt.scatter(observations[:, 0], observations[:, 1], c='b', s=15)
plt.xlim(0., 1.)
plt.ylim(0., 1.)

idxes = reward.argsort(descending=True)

idxes = [idxes[0],idxes[50], idxes[10]]


for idx, c in zip(idxes, ['r', 'y', 'w']):
    print(traj[:, idx, 0])
    print(traj[:, idx, 1])
    plt.scatter(traj[:, idx, 0], traj[:, idx, 1], c=c, s=30)
    line,  = plt.plot(traj[:, idx, 0], traj[:, idx, 1], c=c, linewidth=5)
    line.set_label("R={:.2f}".format(reward[idx].item()))

plt.legend(fontsize=40)
plt.tight_layout()
plt.savefig('plots/reward.png')


plt.clf()

cp = plt.contourf(xbins, ybins, R)
#plt.colorbar(cp) # Add a colorbar to a plot

batch_size = 5000
observations = [env.reset(batch_size=batch_size)]
reward = 0

for i in range(6):
    #action = env.action_space.sample()
    action = torch.rand(batch_size, 2).cuda() * 2 - 1
    obs, r = env.step(action)[:2]
    observations.append(obs)

    reward = reward + r 

traj = torch.stack(observations).detach().cpu().numpy()

plt.xlim(0., 1.)
plt.ylim(0., 1.)

idxes = reward.argsort(descending=False)

exp = torch.exp(reward * 70)
exp = exp/exp.sum()

exp = (exp- exp.min())/exp.max()

import tqdm
for idx, c in tqdm.tqdm(zip(idxes, exp[idxes]), total=len(idxes)):
    plt.scatter(traj[:, idx, 0], traj[:, idx, 1], c=(1., 0., 1., min(float(c * 2.), 1.)), s=100)

plt.tight_layout()
plt.savefig('plots/posterior.png')