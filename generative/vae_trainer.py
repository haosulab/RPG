# trainer to train the generative model from a replay buffer
# we use this to measure the 
import torch
import numpy as np
import tqdm
from tools.config import Configurable
from rpg.buffer import BufferItem
from .vae import DiagGaussian, VAE
from tools.utils import mlp, logger
from tools.nn_base import Network
from tools.optim import LossOptimizer

class IgnoreContext(Network):
    def __init__(self, main, cfg=None):
        super().__init__(cfg)
        self.main = main

    def forward(self, x, context):
        assert context is None
        return self.main(x)


class Trainer(Configurable):
    def __init__(self, env, cfg=None, latent=DiagGaussian.dc, buffer_capcaity=int(1e6), optim_cfg=LossOptimizer.dc,
                    weights=dict(recon=1., kl=1.), batch_size=1000,
                 ) -> None:
        Configurable.__init__(self)

        self.env = env
        self.obs_space = env.observation_space
        self.buffer = BufferItem(buffer_capcaity, {'_shape': self.obs_space.shape})
        self.latent = DiagGaussian(cfg=latent)

        self.make_autoencoder()
        self.vae = VAE(self.encoder, self.decoder, self.latent).to('cuda:0')
        self.optim = LossOptimizer(
            self.vae, cfg=optim_cfg
        )

    def make_autoencoder(self):
        self.encoder = IgnoreContext(mlp(self.obs_space.shape[0], [256, 512, 512, 256], self.latent.get_input_dim()))
        self.decoder = IgnoreContext(mlp(self.latent.embed_dim(), [256, 512, 512, 256], self.obs_space.shape[0]))

        
    def sample_from_buffer(self):
        return self.buffer[self.buffer.sample_idx(self._cfg.batch_size)]
    
    def update(self):
        obs = self.sample_from_buffer()
        out, losses = self.vae(obs, None)
        loss = 0.
        for k, v in losses.items():
            loss += self._cfg.weights.get(k, 1.) * v
        self.optim.optimize(loss)
        logger.logkvs_mean({k: v.item() for k, v in losses.items()})

    @torch.no_grad()
    def visualize_data(self):
        import matplotlib.pyplot as plt
        obs = self.sample_from_buffer()

        recon, _ = self.vae(obs, None)
        with torch.no_grad():
            sampled = self.vae.sample(self._cfg.batch_size, None)
        return {
            'obs': obs,
            'recon': recon,
            'sample': sampled
        }
        

class RandomTrainer(Trainer):
    def __init__(self, env, cfg=None, wandb=None, path='tmp') -> None:
        super().__init__(env, cfg)

    def start(self):
        format_strs = ["stdout", "log", "csv", 'tensorboard']
        kwargs = {}
        if self._cfg.wandb is not None:
            wandb_cfg = dict(self._cfg.wandb)
            if 'stop' not in wandb_cfg:
                format_strs = format_strs[:3] + ['wandb']
                kwargs['config'] = self._cfg
                kwargs['project'] = wandb_cfg.get('project', 'stategen')
                name = None
                kwargs['name'] = wandb_cfg.get('name', None) + (('_' + name) if name is not None else '')
        logger.configure(dir=self._cfg.path, format_strs=format_strs, **kwargs)


        observations = []
        observations.append(self.env.start()[0])
        batch_size = observations[0].shape[0]
        for i in tqdm.trange(100000//batch_size):
            act = np.array([self.env.action_space.sample() for j in range(batch_size)])
            observations.append(self.env.step(act)['next_obs'])

        obs = torch.cat(observations, axis=0)
        self.buffer.append(obs)

        for i in range(1000):
            for j in tqdm.trange(1000):
                self.update()

            trajs =  self.visualize_data() # learned trajectories.

            img1 = self.env.render_traj(trajs)
            img2 = self.env.render_traj({'obs': trajs['recon']})
            img3 = self.env.render_traj({'obs': trajs['sample']})
            img = np.concatenate([img1, img2, img3], axis=1)
            logger.savefig('x.png', img)
            logger.dumpkvs()