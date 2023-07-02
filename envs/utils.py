import numpy as np


def float2array(x, dim):
    if isinstance(x, np.ndarray) or isinstance(x, list):
        x = np.asarray(x)
        assert x.shape[-1] == dim
        return x
    else:
        return np.zeros(dim) + x


def count_occupancy(state, low, high, gap=None, n_bin=None):
    dim = state.shape[-1]
    assert gap is None or n_bin is None

    low = float2array(low, dim)
    high = float2array(high, dim)

    if gap is not None:
        gap = float2array(gap, dim)
        n_bin = np.int32(np.ceil((high - low) / gap))
    else:
        n_bin = np.int32(float2array(n_bin, dim))
        gap = (high - low) / n_bin


    state = state.reshape(-1, dim)
    index = np.int32(np.round((((state - low) / (high - low)) * n_bin).clip(0, n_bin-1)))

    #index = np.prod(index, -1)
    total = np.prod(n_bin)
    ind = 0
    for i in range(dim):
        ind = ind * n_bin[i] + index[..., i]

    out = np.zeros(total, dtype=np.int32)
    unique, cc = np.unique(ind, return_counts=True)
    out[unique] = cc
    return out

def test_count_occupancy():
    x = np.array([
        [0.1, 0.3, 0.5],
        [0.7, 0.8, 0.5],
        [0.13, 0.8, 0.2],
        [0.8, 0.3, 0.2],
    ])

    print(count_occupancy(x, 0, 1, 0.1))


class Embedder:
    # https://github.com/yenchenlin/nerf-pytorch/blob/master/run_nerf_helpers.py
    def __init__(self, **kwargs):
        import torch
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
        import torch
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0, **kwargs):
    if i == -1:
        from torch import nn
        return nn.Identity(), 2
    import torch
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 2,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    embed_kwargs.update(kwargs)
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

def extract_obs_from_tarj(traj):
    if isinstance(traj, dict):
        obs = traj['next_obs']
        import torch
        if isinstance(obs, torch.Tensor):
            obs = obs.detach().cpu().numpy()
    else:
        obs = traj.get_tensor('next_obs', device='numpy')
    return obs

def symlog(x):
    return np.sign(x) * np.log(np.abs(x) + 1)

class EmbedderNP:
    def __init__(self, inp_dim, multires, include_input=True) -> None:
        max_freq = multires - 1
        N_freqs = multires
        self.freq_bands = 2. ** np.linspace(0., max_freq, num=N_freqs)
        self.out_dim = inp_dim * (int(include_input) + multires * 2)
        self.include_input = include_input

    def __call__(self, x):
        inp = x
        x = inp[..., None, :] * self.freq_bands[..., :, None] # freq, d
        x = np.stack((np.sin(x), np.cos(x)), -2)
        assert x.shape[-2] == 2
        x = x.reshape(x.shape[:-3] + (-1,)) # freq, (sin, cos), d
        #print('xx', x.max(), inp.max(), inp.shape)
        if self.include_input:
            return np.concatenate((inp, x), -1)
        else:
            return inp


def get_embeder_np(multires, dim, include_input=True):
    embeder = EmbedderNP(dim, multires, include_input=include_input)
    return embeder, embeder.out_dim

def test_embedder_np():
    import torch
    e1, d1 = get_embedder(10, input_dims=3)
    e2, d2 = get_embeder_np(10, 3)
    #assert d2 == d1

    data = torch.randn(100, 3)

    out1 = e1(data)
    out2 = torch.tensor(e2(data.detach().cpu().numpy())).float()
    assert torch.allclose(out1, out2)

    
def update_occupancy_with_history(occ, history, method='sum'):
    if isinstance(occ, dict):
        return {
            k: update_occupancy_with_history(v, history[k] if history is not None else None) 
            for k, v in occ.items()
        }
    if history is None:
        out = occ 
    elif method == 'sum':
        out = occ + history
    else:
        raise NotImplementedError
    return out 

    
if __name__ == '__main__':
    #test_count_occupancy()
    test_embedder_np()