import torch
import numpy as np
from torch import nn
from tools.nn_base import Network
from tools.optim import OptimModule
from tools.utils import RunningMeanStd, batch_input
#from ..networks import concat
from tools.nn_base import concat

class RNDNet(Network):
    def __init__(self, inp_dim, cfg=None, n_layers=3, dim=512):
        super(RNDNet, self).__init__()
        self.inp_dim = inp_dim
        layers = []
        for i in range(n_layers):
            if i > 0:
                layers.append(nn.LeakyReLU())
            layers.append(nn.Linear(inp_dim, dim))
            inp_dim = dim
        self.main = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.main(x)



# Positional encoding (section 5.1)
class Embedder:
    # https://github.com/yenchenlin/nerf-pytorch/blob/master/run_nerf_helpers.py
    def __init__(self, **kwargs):
        self.kwargs = kwargs

        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)




def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 2,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim




class RNDOptim(OptimModule):
    KEYS = ['obs']
    name = 'rnd'

    def __init__(self, inp_dim, cfg=None, normalizer=False, visit_resolution=0, use_embed=0):
        from .rnd import RNDNet

        self.inp_dim = inp_dim
        if use_embed > 0:
            self.embeder, self.inp_dim = get_embedder(use_embed)
        else:
            self.embeder = None

        network = RNDNet(self.inp_dim)

        super().__init__(network, cfg)
        self.target = RNDNet(network.inp_dim, cfg=network._cfg)
        for param in self.target.parameters():
            param.requires_grad = False
        
        self.rew_rms = RunningMeanStd(last_dim=False)
        self.obs_norm = RunningMeanStd()

        if visit_resolution > 0:
            self.visit_resolution = visit_resolution
            self.counter = np.zeros(shape=(visit_resolution, visit_resolution), dtype=np.int32)
        else:
            self.counter = None



    def forward_network(self, observations, update_norm=True):
        # assert len(observations) == 1 and isinstance(observations, list)
        return self.compute_loss(observations, update_norm=update_norm, reduce=False)

    def compute_loss(self, obs, update_norm=False, reduce=True):
        observations = obs
        #pass

        inps = []
        for i in observations:
            if isinstance(i, dict):
                agent = i['agent'].reshape(-1)
                shape_mean = i['pointcloud'].mean(axis=0)[:3]
                goal_mean = i['pointcloud'].mean(axis=0)[:3]
                outs = concat((agent, shape_mean, goal_mean))
                inps.append(batch_input(outs, self.device))
            else:
                inps.append(batch_input(i, self.device))

                if self.counter is not None and update_norm:
                    x = int(max(min(i[0] * self.visit_resolution, self.visit_resolution-1), 0))
                    y = int(max(min(i[1] * self.visit_resolution, self.visit_resolution-1), 0))
                    self.counter[y, x] += 1

        inps = torch.stack(inps)
        if update_norm:
            self.obs_norm.update(inps.detach().cpu().numpy())
            #raise NotImplementedError("optimize it")
            # TODO: optimize it

        obs_mean = torch.tensor(self.obs_norm.mean, dtype=torch.float32, device=self.device)
        obs_std = torch.tensor(self.obs_norm.std, dtype=torch.float32, device=self.device).clamp(1e-8, np.inf)
        inps = (inps - obs_mean) / obs_std
        inps = inps.clamp(-10., 10.)

        if self.embeder is not None:
            inps = self.embeder(inps)

        predict = self.network(inps)
        with torch.no_grad():
            target =self.target(inps)

        loss = ((predict - target)**2).sum(axis=-1)
        assert loss.dim() == 1
        if reduce:
            loss = loss.mean()
        return loss


    def plot2d(self):
        images = []
        for i in range(32):
            coords = (np.stack([np.arange(32), np.zeros(32)+i], axis=1) + 0.5)/32.
            out = self.forward_network(coords, update_norm=False)
            images.append(out.detach().cpu().numpy())
        return np.array(images)