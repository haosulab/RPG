import numpy as np
import torch
from typing import List, Dict, Any
from .utils import iter_batch
from tools.utils import totensor
from .utils import minibatch_gen

class DataBuffer(dict):
    def loop_over(self, batch_size, keys=None):
        # TODO: preserve the trajectories if necessay.
        if keys is not None:
            return DataBuffer(**{key: self[key] for key in keys}).loop_over(batch_size)

        timesteps = len(self['obs'])
        nenvs = len(self['obs'][0])

        import numpy as np
        index = np.stack(np.meshgrid(np.arange(timesteps), np.arange(nenvs)), axis=-1).reshape(-1, 2)
        return minibatch_gen(self, index, batch_size)


# dataset 
class Trajectory:
    def __init__(self, transitions: List[Dict], nenv, timesteps) -> None:
        self.traj = transitions
        self.nenv = nenv
        self.timesteps = timesteps
        assert len(self.traj) == self.timesteps

        self.index = np.array([(i, j) for j in range(nenv) for i in range(timesteps)])

    def __add__(self, other: "Trajectory"):
        assert self.nenv == other.nenv
        return Trajectory(self.traj + other.traj, self.nenv, self.timesteps + other.timesteps)

    def predict_value(self, key, network, batch_size, index=None, vpred=None):
        if isinstance(key, str):
            key = [key]

        if index is None:
            index = self.index

        for ind in iter_batch(index, batch_size):
            # ignore None
            obs = [[self.traj[i][k][j] if self.traj[i][k] is not None else None for i, j in ind] for k in key]
            value = network(*obs)

            if vpred is None:
                vpred = torch.zeros((self.timesteps, self.nenv,
                    *value.shape[1:]), device=value.device, dtype=value.dtype)

            ind = totensor(ind, dtype=torch.long, device=value.device)
            vpred[ind[:, 0], ind[:, 1]] = value
        return vpred

    def predict_next(self, key, network, batch_size, vpred):
        next_vpred = torch.zeros_like(vpred)
        next_vpred[:-1] = vpred[1:]
        ind = self.get_truncated_index(include_done=True)
        self.predict_value(key, network, batch_size, index=ind, vpred=next_vpred)
        return next_vpred


    def get_tensor(self, key, device='cuda:0', dtype=torch.float32) -> torch.Tensor:
        from tools.utils import totensor, dstack
        if isinstance(self.traj[0][key][0], dict):
            out = {}
            for k in self.traj[0][key][0]:
                value = [[j[k] for j in i[key]] for i in self.traj]
                out[k] = totensor(value, device='cpu', dtype=dtype)
            return out
        else:
            array = [i[key] for i in self.traj]
            if device == 'numpy':
                return np.array(array)
            return totensor(array, device=device, dtype=dtype)

    def get_truncated_done(self, device='cuda:0'):
        # done means that we should ignore the next_value in the end
        # truncated means that we should even ignore rewards in the end
        # truncated must be done ..

        done = self.get_tensor('done', device)
        truncated = done.clone() # done must be truncated ..
        ind = self.get_truncated_index(include_done=True)
        truncated[ind[:, 0], ind[:, 1]] = True
        return done, truncated

    def get_list_by_keys(self, keys) -> DataBuffer:
        return DataBuffer(**{key: [i[key] for i in self.traj] for key in keys})

    def get_truncated_index(self, include_done=False) -> np.ndarray:
        ind = []
        for j in range(self.timesteps):
            for i in range(self.nenv):
                if self.traj[j]['truncated'][i] or j == self.timesteps -1 or (include_done and self.traj[j]['done'][i]):
                    ind.append((j, i))
        return np.array(ind)

    def summarize_epsidoe_info(self):
        # average additional infos, for example success, if necessary.
        n = 0
        rewards = []
        successes = []
        avg_len = 0
        for i in range(self.timesteps):
            if 'episode' in self.traj[i]:
                for j in self.traj[i]['episode']:
                    n += 1
                    rewards.append(j['reward'])
                    if 'success' in j:
                        successes.append(j['success'])
                    avg_len += j['step']
        if n > 0:
            rewards = np.array(rewards)
            out =  {
                "num_episode": n,
                "rewards": rewards.mean(),
                "rewards_std": rewards.std(),
                "avg_len": avg_len / n,
            }
            if len(successes) > 0:
                out['success'] = np.array(successes).mean()
            return out
        else:
            return {"num_episode": n}
        
    @property
    def n(self):
        return self.timesteps * self.nenv