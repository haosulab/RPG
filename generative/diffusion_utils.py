# should copy https://github.com/ermongroup/ddim/blob/main/models/diffusion.py
import torch
import math
from typing import Union
from tools.nn_base import Network
# from diffusers import DDIMScheduler
import torch.nn.functional as F
from einops import reduce


#def gaussian_kl(mu, logvar):
#    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
#    return kl.reshape(mu.shape[0], -1).mean(-1).mean()
def gaussian_kl(mu, logsigma):
    kl = -0.5 * (1 + 2 * logsigma - mu.pow(2) - logsigma.exp().pow(2))
    #return kl.reshape(mu.shape[0], -1).mean(-1)
    return reduce(kl, '... c -> ...', 'mean')

    
class HiddenDDIM(Network):
    def __init__(
        self,
        unet: torch.nn.Module,
        scheduler,

        cfg=None,
        num_inference_steps=50,
        loss_type='l2',
    ) -> None:
        super().__init__()

        self.scheduler = scheduler
        self.unet = unet
        self.num_train_timesteps = len(scheduler.timesteps)


    def sample(self, init_latents, context, **kwargs):
        self.scheduler.set_timesteps(self._cfg.num_inference_steps)
        latents = init_latents

        for t in self.scheduler.timesteps: # TODO: add progress bar
            noise_pred = self.unet(latents, context, torch.ones(latents.shape[0], device=latents.device) * t) # follow the 
            latents = self.scheduler.step(noise_pred, t, latents, **kwargs).prev_sample # build latents
        return latents

    def sample_init_latents(self, batch_size, context):
        return torch.randn((batch_size, *self.latent_shape), device=self.device)
        
    def forward(self, init_latents, context, t=None):
        batch_size = len(init_latents)
        self.latent_shape = init_latents.shape[1:]

        if t is None:
            t = torch.randint(0, self.num_train_timesteps, (batch_size,), device=self.device).long()

        noise = torch.randn(init_latents.shape, device=self.device)
        noisy_input = self.scheduler.add_noise(init_latents, noise, t)
        noise_pred = self.unet(noisy_input, context, t)

        assert noise_pred.shape == noise.shape

        if self._cfg.loss_type == 'l1':
            loss = F.l1_loss(noise, noise_pred)
        elif self._cfg.loss_type == 'l2':
            loss = F.mse_loss(noise, noise_pred)
        else:
            raise NotImplementedError

        T = t * 0 + self.num_train_timesteps - 1

        last_mu, last_sigma = self.add_noise(init_latents, T)
        kl_loss = gaussian_kl(last_mu, last_sigma.log())

        return init_latents, {
            'denoise': loss,
            'kl': kl_loss,
        }

    def add_noise(self, original_samples, t):

        ddim = self.scheduler

        sqrt_alpha_prod = ddim.alphas_cumprod[t] ** 0.5
        sqrt_alpha_prod = ddim.match_shape(sqrt_alpha_prod, original_samples)
        sqrt_one_minus_alpha_prod = (1 - ddim.alphas_cumprod[t]) ** 0.5
        sqrt_one_minus_alpha_prod = ddim.match_shape(sqrt_one_minus_alpha_prod, original_samples)

        return sqrt_alpha_prod * original_samples, sqrt_one_minus_alpha_prod