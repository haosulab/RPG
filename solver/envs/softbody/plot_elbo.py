import torch
import numpy as np
import matplotlib.pyplot as plt
from solver.envs.softbody.plot_envs import TripleCont


env = TripleCont()
rpg = torch.load('exp/plot_env/rpg2/model_20')

data = rpg.inference(env, batch_size=10000)
print(data.keys())

z = data['z']


# plt.su

from tools.utils import tonumpy
from sklearn.decomposition import PCA

pca = PCA(n_components=2) # to 3 dimension

embedding = tonumpy(z)
#shape = embedding.shape
#embedding = embedding.reshape(-1, shape[-1])
#components = pca.fit_transform(embedding)
components = embedding.reshape(-1, 2)

# assert np.allclose(components, pca.transform(embedding))


nbins = 100
xmin, xmax = 0., 1.0
ymin, ymax = 0., 1.0
xbins = np.linspace(xmin, xmax, nbins)
ybins = np.linspace(ymin, ymax, nbins)


from tools.utils import totensor
pp = totensor(np.meshgrid(xbins, ybins), device='cuda:0').permute(1, 2, 0)
r =  -(pp[:, :, None, :2] - env.goals[None, None, :, :2]).norm(dim=-1).min(dim=-1)[0]


fig = plt.figure()
pp = pp.detach().cpu().numpy()
R =  ((r - r.min())/ (r.max() - r.min())).detach().cpu().numpy()

cp = plt.contourf(xbins, ybins, R)
# ax = plt.axes(projection='3d')
# ax.contour3D(pp[:, :, 0], pp[:, :, 1], R, 50, cmap='viridis', edgecolor='none')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# ax.view_init(60, 35)
# plt.imshow(R)

fig.colorbar(cp) # Add a colorbar to a plot
plt.savefig('exp/plot_env/reward.png')
plt.clf()


def compute_density(states):
    states = states.reshape(-1, states.shape[-1])
    states = states[:, :2].detach().cpu().numpy() + np.random.normal(0, 0.03, (len(states), 2))
    #xmin, xmax = states[:, 0].min(), states[:, 0].max()
    #ymin, ymax = states[:, 1].min(), states[:, 1].max()
    density, _, _ = np.histogram2d(states[:, 0], states[:, 1], bins=[xbins, ybins])
    return density.transpose(1, 0)

print(data['s'].shape)
density = compute_density(data['s'][1:])[::-1]
#from IPython import embed; embed()

idx = data['r'][-1, :].argmax()
print('idx', idx)

traj = data['s'][:, idx].detach().cpu().numpy()
plt.imshow((density/min(density.max(), 1000.)).clip(0., 1.), extent=[0., 1., 0., 1.])
plt.scatter(traj[:, 0], traj[:, 1], c='r', s=15)
plt.xlim(0., 1.)
plt.ylim(0., 1.)
plt.savefig('exp/plot_env/density.png')
plt.clf()


s = totensor(traj[:, None], device='cuda:0')
#print(s.shape, data['a'][:, :1].shape)
#exit(0)
dist_z = rpg.info_log_q.main((s, data['a'][:, :1]*0))
mean = dist_z.dist.loc.detach()[-1, 0]
scale = dist_z.dist.scale.detach()[-1, 0]

flatten_z = data['z'].reshape(-1, data['z'].shape[-1])
flatten_z = flatten_z.detach().cpu().numpy()

d = np.exp((-(flatten_z - mean.detach().cpu().numpy())**2 / scale.detach().cpu().numpy()**2).sum(-1)/10.)


plt.figure(figsize=(10, 10))
color = np.array([0., 0., 0.1, 1.])[None, :].repeat(components.shape[0], axis=0)
color[:, 1] = (d/d.max()).clip(0., 1.)

plt.scatter(components[:, 0], components[:, 1], c=color, s=1)
plt.scatter(components[idx:(idx+1), 0], components[idx:(idx+1), 1], c='r', s=30)
#plt.xlim(-3.0, 3.0)
#plt.ylim(-3.0, 3.0)
plt.savefig('exp/plot_env/z.png')
plt.clf()