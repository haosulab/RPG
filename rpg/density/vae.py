import torch
from torch import nn 
from .density_estimator import DensityEstimator
from tools.nn_base import Network
from tools.utils import logger
from generative.vae import VAE as VAENet, DiagGaussian
from tools.utils import mlp, logger


class IgnoreContext(Network):
    def __init__(self, main, cfg=None):
        super().__init__(cfg)
        self.main = main

    def forward(self, x, context):
        assert context is None
        return self.main(x)


class VAE(DensityEstimator):
    def __init__(self, space, cfg=None, normalizer='none', latent=DiagGaussian.dc, beta=1.) -> None:
        super().__init__(space, cfg, normalizer)
        self.beta = beta

    def make_network(self, space):
        assert len(space.shape) == 1
        latent = DiagGaussian(cfg=self._cfg.latent)
        encoder = IgnoreContext(mlp(space.shape[0], [256, 512, 512, 256], latent.get_input_dim()))
        decoder = IgnoreContext(mlp(latent.embed_dim(), [256, 512, 512, 256], space.shape[0]))
        return VAENet(encoder, decoder, latent)

    def compute_loss(self, samples, log=False):
        out, losses = self.network(samples, None)

        if log:
            logger.logkvs_mean({self.name + '_' + k: v.mean().item() for k, v in losses.items()})

        loss = losses['recon'] + self.beta * losses['kl']
        return loss[..., None]

    def _update(self, samples):
        loss = self.compute_loss(samples, log=True)
        self.optimize(loss.mean())
        return -loss.detach()

    def _log_prob(self, samples):
        return -self.compute_loss(samples) # negative as the log prob ..