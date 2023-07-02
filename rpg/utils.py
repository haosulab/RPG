import tqdm
import numpy as np
import torch
from nn.space import Discrete, Box, MixtureSpace
from tools.config import Configurable



def flatten_obs(obs_seq):
    if isinstance(obs_seq, torch.Tensor):
        next_obs = obs_seq.reshape(-1, *obs_seq.shape[2:])
    else:
        assert isinstance(obs_seq[0], dict)
        next_obs = {}
        for k in obs_seq[0]:
            # [T, B, ...]
            next_obs[k] = torch.stack([v[k] for v in obs_seq])
        for k, v in next_obs.items():
                next_obs[k] = v.reshape(-1, *v.shape[2:])
    return next_obs

def iter_batch(index, batch_size):
    return np.array_split(np.array(index), max(len(index)//batch_size, 1))


def minibatch_gen(traj, index, batch_size, KEY_LIST=None, verbose=False):
    # traj is dict of [nsteps, nenv, datas.]
    if KEY_LIST is None:
        KEY_LIST = list(traj.keys())
    tmp = iter_batch(np.random.permutation(len(index)), batch_size)
    if verbose:
        tmp = tqdm.tqdm(tmp, total=len(tmp))
    for idx in tmp:
        k = index[idx]
        yield {
            key: [traj[key][i][j] for i, j in k] if traj[key] is not None else None
            for key in KEY_LIST
        }

def create_hidden_space(z_dim, z_cont_dim):
    if z_cont_dim == 0:
        z_space = Discrete(z_dim)
        z_space.inp_shape = (z_dim,)
    elif z_dim == 0:
        z_space = Box(-1, 1, (z_cont_dim,))
        z_space.inp_shape = (z_cont_dim,)
    else:
        z_space = MixtureSpace(Discrete(z_dim), Box(-1, 1, (z_cont_dim,)))
        z_space.inp_shape = (z_dim + z_cont_dim,)
        raise NotImplementedError("The Z Transform is not implemented ..")
    return z_space

import torch
class ZTransform(torch.nn.Module):
    def __init__(self, z_space) -> None:
        super().__init__()
        self.z_space = z_space
        if isinstance(self.z_space, Discrete):
            self.output_dim = self.z_space.n
        else:
            self.output_dim = self.z_space.shape[-1]

    def forward(self, x):
        if isinstance(self.z_space, Discrete):
            assert x.max() < self.z_space.n, f"{x.max()} < {self.z_space.n}"
            return torch.nn.functional.one_hot(x, self.z_space.n).float()
        else:
            return x


def config_hidden(hidden, hidden_space):
    from gym.spaces import Box, Discrete as Categorical
    from tools.config import merge_inputs, CN
    if isinstance(hidden_space, Box):
        default_hidden_head = dict(TYPE='Normal', linear=True, std_scale=0.2)
    elif isinstance(hidden_space, Categorical):
        default_hidden_head = dict(TYPE='Discrete')
    else:
        raise NotImplementedError
    if hidden is not None:
        hidden_head = merge_inputs(CN(default_hidden_head), **hidden)
    else:
        hidden_head = default_hidden_head
    return hidden_head


# def done_rewards_values(values, prefix, dones):
#     if dones is not None:
#         assert dones.shape == prefix.shape
#         not_done = 1 - dones

#         alive = torch.cumprod(not_done, 0)
#         assert (alive <= 1.).all()

#         r = prefix.clone()
#         r[1:] = r[1:] - r[:-1] # get the gamma decayed rewards ..

#         alive_r = torch.ones_like(alive)
#         alive_r[1:] = alive[:-1]
#         prefix = (r * alive_r).cumsum(0)

#         values = values * alive
#         from tools.utils import logger
#         logger.logkv_mean('not_done',  not_done.mean().item())
#     return values, prefix


def lmbda_decay_weight(lmbda, horizon, lmbda_last):
    weights = []
    sum_lmbda = 0.
    for _ in range(horizon):
        weights.append(lmbda)
        sum_lmbda += lmbda
        lmbda *= lmbda
    weights = torch.tensor(weights, dtype=torch.float32)
    if lmbda_last:
        weights[-1] += (1./(1-lmbda) - sum_lmbda) 
    weights = weights / weights.sum()
    return weights

def compute_value_prefix(rewards, gamma):
    v = 0
    discount = 1
    value_prefix = []
    for i in range(len(rewards)):
        v = v + rewards[i] * discount
        value_prefix.append(v)
        discount = discount * gamma
    return torch.stack(value_prefix)

def masked_temporal_mse(a, b, mask):
    assert a.shape[:-1] == b.shape[:-1], f'{a.shape} vs {b.shape}'
    if a.shape[-1] != b.shape[-1]:
        assert b.shape[-1] == 1 and (a.shape[-1] in [1, 2]), f'{a.shape} vs {b.shape}'
    difference = ((a-b)**2).mean(axis=-1)
    assert difference.shape == mask.shape
    return (difference * mask).sum(axis=0)


def compute_gae_by_hand(reward, value, next_value, done, truncated, gamma, lmbda, mode='approx', return_sum_weight_value=False):

    reward = reward.to(torch.float64)
    if value is not None:
        value = value.to(torch.float64)
    next_value = next_value.to(torch.float64)
    # follow https://arxiv.org/pdf/1506.02438.pdf
    if mode != 'exact':
        assert not return_sum_weight_value
        import tqdm
        gae = []
        for i in tqdm.trange(len(reward)):
            adv = 0.
            # legacy ..

            if mode == 'approx':
                for j in range(i, len(reward)):
                    delta = reward[j] + next_value[j] * gamma - value[j]
                    adv += (gamma * lmbda)**(j-i) * delta * (1. - done[j].float())[..., None]
            elif mode == 'slow':
                R = 0
                discount_gamma = 1.
                discount_lmbda = 1.

                lmbda_sum = 0.
                not_truncated = 1.0
                lastA = 0.
                for j in range(i, len(reward)):

                    R = R + reward[j] * discount_gamma

                    mask_done = (1. - done[j].float())[..., None]
                    A = R + (discount_gamma * gamma) * next_value[j] * mask_done - value[i] # done only stop future rewards ..

                    lmbda_sum += discount_lmbda

                    lastA = A * not_truncated + (1-not_truncated) * lastA
                    adv += (A * discount_lmbda) 

                    mask_truncated = (1. - truncated[j].float())[..., None] # mask truncated will stop future computation.

                    discount_gamma = discount_gamma * mask_truncated
                    discount_lmbda = discount_lmbda * mask_truncated
                    not_truncated = not_truncated * mask_truncated

                    # note that we will count done; always ...

                    discount_gamma = discount_gamma * gamma
                    discount_lmbda = discount_lmbda * lmbda

                #adv = adv/ lmbda_sum # normalize it based on the final result.
                adv = (adv + lastA  * (1./ (1.-lmbda) - lmbda_sum)) * (1-lmbda)
            else:
                raise NotImplementedError

            gae.append(adv)
    else:
        """
        1               -V(s_t)  + r_t                                                                                     + gamma * V(s_{t+1})
        lmabda          -V(s_t)  + r_t + gamma * r_{t+1}                                                                   + gamma^2 * V(s_{t+2})
        lambda^2        -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2}                                               + ...
        lambda^3        -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + gamma^3 * r_{t+3}

        We then normalize it by the sum of the lambda^i
        """
        sum_lambda = 0.
        sum_reward = 0.
        sum_end_v = 0.
        last_value = 0.
        gae = []
        mask_done = (1. - done.float())[..., None]
        mask_truncated = (1 - truncated.float())[..., None]
        if return_sum_weight_value:
            sum_weights = []
            last_values = []
            total = []

        for i in reversed(range(len(reward))):
            sum_lambda = sum_lambda * mask_truncated[i]
            sum_reward = sum_reward * mask_truncated[i]
            sum_end_v = sum_end_v * mask_truncated[i]

            sum_lambda = 1. + lmbda * sum_lambda
            sum_reward = lmbda * gamma * sum_reward + sum_lambda * reward[i]

            next_v = next_value[i] * mask_done[i]
            sum_end_v =  lmbda * gamma * sum_end_v  + gamma * next_v

            last_value = last_value * mask_truncated[i] + next_v * (1-mask_truncated[i]) # if truncated.. use the next_value; other wise..
            last_value = last_value * gamma + reward[i]
            # if i == len(reward) - 1:
            #     print('during the last', sum_reward, gamma, next_value[i], mask_done[i], value[i])
            sumA = sum_reward + sum_end_v

            if return_sum_weight_value:
                sum_weights.append(sum_lambda)
                last_values.append(last_value)
                total.append(sumA)

            expected_value = (sumA + last_value  * (1./ (1.-lmbda) - sum_lambda)) * (1-lmbda)
            # gg = sumA / sum_lambda 
            gae.append(expected_value - (value[i] if value is not None else 0))

        if return_sum_weight_value:
            sum_weights = torch.stack(sum_weights[::-1])
            last_values = torch.stack(last_values[::-1])
            total = torch.stack(total[::-1])
            return total, sum_weights, last_values

        gae = gae[::-1]

    return torch.stack(gae).float() #* (1-lmbda) 


import math
def positional_encoding(dim, position):
    # """
    # :param d_model: dimension of the model
    # :param length: length of positions
    # :return: length*d_model position matrix
    # """
    if dim == 1:
        return position
    import torch
    assert dim % 2 == 0
    dim = dim // 2
    freq_bands = torch.linspace(0, dim-1, dim, device=position.device)
    x = position[..., None, :] * freq_bands[..., :, None]
    x = x.view(x.shape[:-2] + (-1,))
    return torch.cat([torch.sin(x), torch.cos(x), position], dim=-1)

  
class Embedder(torch.nn.Module):
    def __init__(self, input_dims=2, multires=4, include_input=True, identity_scale=0.001):
        super().__init__()

        embed_fns = []
        #d = self.kwargs['input_dims']
        self.multires = multires
        self.input_dims = input_dims
        self.identity_scale = identity_scale 

        if multires > 0:
            d = input_dims
            out_dim = 0

            if include_input:
                embed_fns.append(lambda x : x * identity_scale)
                out_dim += d
                
            max_freq_log2 = multires - 1
            num_freqs = multires
            log_sampling = True
            periodic_fns =  [torch.sin, torch.cos]

            #max_freq = self.kwargs['max_freq_log2']
            max_freq = max_freq_log2
            N_freqs = num_freqs
            # N_freqs = self.kwargs['num_freqs']
            
            if log_sampling:
                freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
            else:
                freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
                
            for freq in freq_bands:
                for p_fn in periodic_fns:
                    embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                    out_dim += d
                        
            self.embed_fns = embed_fns
            self.out_dim = out_dim
        else:
            self.out_dim = input_dims
        
    def forward(self, inputs):
        if self.multires == 0:
            return inputs
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

    def decode(self, outputs):
        if self.multires == 0:
            return outputs
        return outputs[..., :self.input_dims] / self.identity_scale
 