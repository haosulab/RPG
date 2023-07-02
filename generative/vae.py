import torch
from tools.nn_base import Network
from typing import Union
from tools.nn_base import Network
# from diffusers import DDPMScheduler, DDIMScheduler
import torch.nn.functional as F
from .diffusion_utils import HiddenDDIM, gaussian_kl
from tools.config import Configurable
from einops import rearrange, reduce


class DiagGaussian(Configurable, torch.nn.Module):
    def __init__(self, cfg=None, latent_dim=16):
        Configurable.__init__(self)
        torch.nn.Module.__init__(self)

        self.device = 'cuda:0'
        self.latent_dim = latent_dim

    def sample(self, latents, context):
        return latents

    def sample_init_latents(self, batch_size, context):
        return torch.randn((batch_size, self.latent_dim), device=self.device)

    def __call__(self, latent_input, context):
        #raise NotImplementedError
        mu, log_sigma = torch.chunk(latent_input, 2, dim=-1)
        sigma = log_sigma.exp()
        #self.latent_shape = mu.shape[1:]
        return torch.randn_like(mu) * sigma + mu,   {'kl': gaussian_kl(mu, log_sigma)}

    def get_input_dim(self):
        return self.latent_dim * 2

    def embed_dim(self):
        return self.latent_dim




class VAE(Network):
    def __init__(self, encoder: Network, decoder: Network, latent: Union[HiddenDDIM, DiagGaussian], cfg=None) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent = latent

    def sample(self, batch_size, context):
        assert context is None or context.shape[0] == batch_size
        init_latent = self.latent.sample_init_latents(batch_size, context)
        return self.decoder(self.latent.sample(init_latent, context), context)
    
    def forward(self, input, context):
        latent_input = self.encoder(input, context)
        latent, losses = self.latent(latent_input, context)

        decoded = self.decoder(latent, context)
        assert decoded.shape == input.shape
        losses['recon'] = ((decoded - input) ** 2)

        losses['recon'] = reduce(losses['recon'], '... c -> ...', 'mean')
        return decoded, losses