import tqdm
import matplotlib.pyplot as plt
import gym
import torch
import numpy as np  
import torch_scatter
from tools.utils import logger
from tools.utils import totensor
from tools.config import Configurable
from rpg.density import DensityEstimator
from gym.spaces import Box
from rpg.utils import Embedder

"""
# supervised learning for density measurement
# see notion here: https://www.notion.so/Density-Measure-fa57fc6eda8c4b94b9a8e37bb1490bd8

Task list
    1. dataset
        - simple Gaussian
        - dataset from rpg (buffer of test_new_triple)
        - ant buffer, and we change the ratio of each grid
    2. evaluation metrics
    3. model and optimizer
        - adhoc: return an occupancy map (with anchor )
 
    4. visualization
"""

class DatasetBase(Configurable):
    def __init__(self, cfg=None) -> None:
        super().__init__()

    def get_obs_space(self):
        raise NotImplementedError

    def sample(self, batch_size):
        # sample dataset
        raise NotImplementedError

    def tokenize(self, inp):
        # return occupancy
        raise NotImplementedError

    def visualize(self, occupancy):
        # visualize the occupancy
        raise NotImplementedError

    def count(self, inp, value, reduce='sum'):
        index, N = self.tokenize(inp)
        if value is None:
            value = torch.ones_like(index) 
        return torch_scatter.scatter(value, index, dim=0, dim_size=N, reduce=reduce)

    def test(self):
        plt.clf()
        data = self.sample(1000)
        occupancy = self.count(data, None)
        self.visualize(occupancy)
        plt.savefig('test.png')
        

class GaussianDataset(DatasetBase):
    def __init__(self, cfg=None, N=100, embed_dim=0) -> None:
        super().__init__()
        self.N = N
        self.bins = np.linspace(-5, 5., self.N)
        self.embedder = Embedder(1, embed_dim)
        self.out_dim = self.embedder.out_dim

    def get_obs_space(self):
        return gym.spaces.Box(-5, 5., shape=(self.out_dim,))

    def sample(self, batch_size):
        n1 = int(batch_size * 0.3)
        X = np.concatenate(
            (np.random.normal(-1, 1, n1), np.random.normal(3, 0.3, batch_size - n1))
        )[:, np.newaxis]

        inp = totensor(X, device='cuda:0').clip(-5, 4.999999)
        output = self.embedder(inp / 5)
        assert torch.allclose(inp, self.embedder.decode(output) * 5)
        return output

    def tokenize(self, inp):
        inp = self.embedder.decode(inp) * 5
        inp = inp.reshape(-1)
        # print(plt.hist(inp.cpu().numpy(), bins=100)[1].shape)
        return ((inp / 10 + 0.5) * (self.N-1)).long(), self.N


    def visualize(self, occupancy):
        #return super().visualize(occupancy)
        occupancy = occupancy.detach().cpu().numpy()
        bins = np.append(self.bins, self.bins[-1] + 10. / self.N)
        plt.stairs(occupancy, bins, fill=True, label='occupancy')


class Env2DDataset(DatasetBase):
    def __init__(self, cfg=None, path='tmp/new/buffer.pt', N=25) -> None:
        super().__init__()
        with torch.no_grad():
            buffer = torch.load(path)
            self.data = buffer._next_obs[:buffer.total_size()].cpu().numpy()
        self.N = N
        self.xedges = np.linspace(0., 1, self.N + 1)
        self.yedges = np.linspace(0., 1, self.N + 1)

    def get_obs_space(self):
        return gym.spaces.Box(-5, 5., shape=(3,))

    def sample(self, batch_size):
        idx = np.random.choice(self.data.shape[0], batch_size)
        return totensor(self.data[idx], device='cuda:0')

    def tokenize(self, inp):
        inp = inp[..., :2].reshape(-1, 2) #.detach().cpu().numpy()
        # plt.scatter(inp[:, 0], inp[:, 1])
        # plt.savefig('test.png')
        # exit(0)
        # count, xedges, yedges = np.histogram2d(
        #     inp[:, 1], inp[:, 0], bins=self.N, range=[[0., 1], [0., 1]]
        # )
        # #plt.hist2d(inp[:, 0], inp[:, 1], bins=self.N, range=[[-1., 1], [-1., 1]])
        # return totensor(count, device='cuda:0')
        x = (inp[:, 1] * (self.N-1)).long()
        y = (inp[:, 0] * (self.N-1)).long()
        return x * self.N + y, self.N * self.N

    def visualize(self, occupancy):
        count2d = occupancy.detach().cpu().numpy().reshape(self.N, self.N)
        plt.pcolormesh(self.xedges, self.yedges, count2d, shading='auto')
        plt.colorbar()


def make_dataset(dataset_name, env_cfg=None):
    if dataset_name == 'twonormal':
        return GaussianDataset(cfg=env_cfg)
    elif dataset_name == 'env2d':
        return Env2DDataset(cfg=env_cfg)
    else:
        raise NotADirectoryError


class Trainer(Configurable): 
    def __init__(
        self, cfg=None, dataset_name=None, 
        env_cfg=None,
        density=DensityEstimator.to_build(TYPE="RND"),
        max_epoch=1000,
        batch_size = 256,
        batch_num=2000,
        path = None,
        vis=dict(scale=1.,),
    ) -> None:
        super().__init__()
        self.dataset = make_dataset(dataset_name, env_cfg)
        self.density: DensityEstimator = DensityEstimator.build(self.dataset.get_obs_space(), cfg=density).cuda()
        self.density.register_discretizer(self.dataset.tokenize)

        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.batch_num = batch_num

        # TODO make logger with wandb easier to config ..
        logger.configure(dir=path, format_strs=['stdout', 'log', 'csv'])

    def test_dataset(self):
        self.dataset.test()

    def train(self):
        for i in tqdm.trange(self.max_epoch):
            for j in tqdm.trange(self.batch_num):
                data = self.dataset.sample(self.batch_size)
                self.density.update(data)

            vis = self._cfg.vis
            with torch.no_grad():
                data = self.dataset.sample(1000)
                log_prob = self.density.log_prob(data)

                #countprob = torch.softmax(log_prob[..., 0] * vis.scale, dim=0)
                prob = (log_prob[..., 0] * vis.scale).exp()
                value = self.dataset.count(data, prob, reduce='mean')
                plt.clf()
                self.dataset.visualize(value)
                logger.savefig('log_prob_avg.png')


if __name__ == '__main__':
    trainer = Trainer.parse(dataset_name='twonormal')
    trainer.train()