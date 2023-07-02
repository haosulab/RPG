import torch
import gym
import numpy as np
from tools.config import Configurable

class BufferItem:
    def __init__(self, capacity: int, space: dict) -> None:
        self._capacity = capacity
        self._space = space
        if '_shape' not in space:
            self._data = {k: BufferItem(capacity, v) for k, v in space.items()}
        else:
            self._data = torch.empty((capacity, *space['_shape']), dtype=space.get('_dtype', torch.float32), device=space.get('_device', 'cuda:0'))
            self._full = False
            self.idx = 0

    def __getitem__(self, key):
        if isinstance(self._data, dict):
            return {k: v[key] for k, v in self._data.items()}
        else:
            return self._data[key]

    def append(self, data):
        if isinstance(self._data, dict):
            for k, v in data.items():
                self._data[k].append(v)
        else:
            assert len(data.shape) > 1
            horizon = len(data)
            end = min(self.idx + horizon, self._capacity)
            # print(self._data.shape, data.shape, end, self.idx, self._capacity)
            self._data[self.idx:end] = data[:end-self.idx]

            new_idx = (self.idx + horizon) % self._capacity
            if self.idx + horizon >= self._capacity:
                self._full = True
                if new_idx != 0:
                    self._data[:new_idx] = data[end-self.idx:]
            self.idx = new_idx

    def __len__(self):
        if isinstance(self._data, dict):
            for k, v in self._data.items():
                return len(v)
        else:
            return self._capacity if self._full else self.idx

    @property
    def device(self):
        if isinstance(self._data, dict):
            for k, v in self._data.items():
                return v.device
        return self._data.device

    def sample_idx(self, batch_size):
        return torch.from_numpy(np.random.choice(len(self), batch_size)).to(self.device)



from typing import Union, Optional
from dataclasses import dataclass

@dataclass
class TrajSeg:
    obs_seq: Union[torch.Tensor, dict]
    timesteps: torch.LongTensor
    action: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    truncated_mask: torch.Tensor
    done: torch.Tensor
    z: Optional[torch.Tensor]
    future: None


def rand(a, b):
    return torch.minimum(torch.floor(torch.rand(b.shape, device=b.device) * (b-a).float()).long() + a, b-1)

class ReplayBuffer(Configurable):
    # replay buffer with done ..
    def __init__(self,
        obs_space, action_space, episode_length, horizon,
        cfg=None, device='cuda:0', max_episode_num=1000,
        max_capacity=800000,
    ):
        super().__init__()

        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.episode_length = episode_length
        self.capacity = min(max_episode_num * episode_length, max_capacity)
        self.horizon = horizon

        dtype = torch.float32
        obs_device = self.device if len(obs_space.shape) == 1 else 'cpu'
        # obs_device = 'cpu'

        obs_dtype = self.obs_dtype = dtype if len(obs_space.shape) == 1 else torch.uint8
        self.obs_device = obs_device

        if not isinstance(obs_space, dict):
            obs_shape = obs_space.shape
            self._obs = torch.empty((self.capacity, *obs_shape), dtype=obs_dtype, device=obs_device) # avoid last buggy..
            self._next_obs = torch.empty((self.capacity, *obs_shape), dtype=obs_dtype, device=obs_device)
            self.obs_shape = obs_shape
        else:
            self._obs = {}
            self._next_obs = {}
            for k, v in obs_space.items():
                self._obs[k] = torch.empty((self.capacity, *v.shape), dtype=dtype, device='cpu') # avoid last buggy..
                self._next_obs[k] = torch.empty((self.capacity, *v.shape), dtype=dtype, device='cpu')


        if isinstance(action_space, gym.spaces.Box):
            self._action = torch.empty((self.capacity, *action_space.shape), dtype=torch.float32, device=self.device)
        else:
            self._action = torch.empty((self.capacity,), dtype=torch.long, device=self.device)


        self._reward = torch.empty((self.capacity, 1), dtype=torch.float32, device=self.device)
        self._dones = torch.empty((self.capacity, 1), dtype=torch.float32, device=self.device)
        self._truncated = torch.empty((self.capacity, 1), dtype=torch.float32, device=self.device)
        self._timesteps = torch.empty((self.capacity,), dtype=torch.long, device=self.device) - 1
        self._step2trajend = torch.empty((self.capacity,), dtype=torch.long, device=self.device)
        self._z = None

        self._eps = 1e-6
        self._full = False
        self.idx = 0

    def total_size(self):
        if self._full:
            return self.capacity
        return self.idx

    @torch.no_grad()
    def add(self, traj):
        from .traj import Trajectory
        traj: Trajectory

        # assert self.episode_length == traj.timesteps, "episode length mismatch"
        length = traj.timesteps
        cur_obs = traj.get_tensor('obs', self.obs_device, dtype=None)
        next_obs = traj.get_tensor('next_obs', self.obs_device, dtype=None)

        actions = traj.get_tensor('a', self.device)
        rewards = traj.get_tensor('r', self.device)
        timesteps = traj.get_tensor('timestep', self.device)
        dones, truncated = traj.get_truncated_done(self.device)
        assert truncated[-1].all()

        z = traj.get_tensor('z', self.device, dtype=None)
        if self._z is None:
            self._z = torch.empty((self.capacity, *z.shape[2:]), dtype=z.dtype, device=self.device)

        for i in range(traj.nenv):
            l = min(length, self.capacity - self.idx)


            if isinstance(self._obs, dict):
                for k, v in self._obs.items():
                    v[self.idx:self.idx+l] = cur_obs[k][:l, i]
                    self._next_obs[k][self.idx:self.idx+l] = next_obs[k][:l, i]
            else:
                self._obs[self.idx:self.idx+l] = cur_obs[:l, i]
                self._next_obs[self.idx:self.idx+l] = next_obs[:l, i]

            self._action[self.idx:self.idx+l] = actions[:l, i]
            self._reward[self.idx:self.idx+l] = rewards[:l, i]
            self._dones[self.idx:self.idx+l] = dones[:l, i, None]

            truncated[l-1, i] = True # last one is always truncated ..
            self._truncated[self.idx:self.idx+l] = truncated[:l, i, None]
            self._timesteps[self.idx:self.idx+l] = timesteps[:l, i]

            d = 0
            for j in range(self.idx + l - 1, self.idx-1, -1):
                if self._truncated[j]:
                    d = 0
                d += 1
                self._step2trajend[j] = d
            # print(z[:l, i])
            # print(dones[:l, i].any())
            # print(truncated[:l, i], d)
            # #self._step2trajend[self.idx: self.idx+l] = torch.arange(l, 0, -1, device=self.device)
            # exit(0)

            if self._z is not None:
                self._z[self.idx:self.idx+l] = z[:l, i]

            self.idx = (self.idx + l) % self.capacity
            self._full = self._full or self.idx == 0

    @torch.no_grad()
    def sample(self, batch_size, horizon=None, latest=None):
        # NOTE that the data after truncated will be something random ..
        horizon = horizon or self.horizon
        total = self.total_size()

        if latest is not None:
            assert latest <= self.capacity
            latest = min(latest, total)
            idxs = np.random.choice(latest, batch_size, replace=(latest < batch_size))
            idxs = (self.idx - latest - idxs + self.capacity) % self.capacity

            idxs = torch.from_numpy(idxs).to(self.device)
        else:
            idxs = torch.from_numpy(np.random.choice(total, batch_size, replace=not self._full)).to(self.device)

        if isinstance(self._obs, dict):
            obs_seq = []
        else:
            obs_seq = torch.empty((horizon + 1, batch_size, *self.obs_shape), dtype=torch.float32, device=self.device)

        timesteps = torch.empty((horizon + 1, batch_size), dtype=torch.long, device=self.device)

        action = torch.empty((horizon, batch_size, *self._action.shape[1:]), dtype=self._action.dtype, device=self.device)
        reward = torch.empty((horizon, batch_size, 1), dtype=torch.float32, device=self.device)
        done = torch.empty((horizon, batch_size, 1), dtype=torch.float32, device=self.device)
        truncated = torch.empty((horizon, batch_size, 1), dtype=torch.float32, device=self.device)

        def get_obs_by_idx(_obs, idxs):
            if isinstance(_obs, dict):
                return {k: v[idxs.cpu()].to(self.device) for k, v in _obs.items()}
            return _obs[idxs.to(self.obs_device)].to(self.device)

        obs = get_obs_by_idx(self._obs, idxs)
        if isinstance(self._obs, dict):
            obs_seq.append(obs)
        else:
            obs_seq[0] = obs
        timesteps[0] = self._timesteps[idxs]

        if self._z is not None:
            z = self._z[idxs] # NOTE: we require the whole z to be the same ..


        for t in range(horizon):
            _idxs = (idxs + t).clamp(0, self.capacity-1)
            action[t] = self._action[_idxs]
            reward[t] = self._reward[_idxs]
            done[t] = self._dones[_idxs]
            truncated[t] = self._truncated[_idxs]

            next_obs = get_obs_by_idx(self._next_obs, _idxs)
            if isinstance(self._obs, dict):
                obs_seq.append(next_obs)
            else:
                obs_seq[t+1] = next_obs #self._next_obs[_idxs]
            timesteps[t+1] = self._timesteps[_idxs] + 1


        truncated_mask = torch.ones_like(truncated) # we weill not predict state after done ..
        truncated_mask[1:] = 1 - (truncated.cumsum(0)[:-1] > 0).float()


        output = TrajSeg(obs_seq, timesteps, action, reward, done, truncated_mask[..., 0], None, None)

        if self._z is not None:
            output.z = z
            #output += (z,) #NOTE: this is the z before the state ..

            a = idxs
            b = idxs + self._step2trajend[idxs]

            idx_future = rand(a, b)

            # c = torch.where(self._z[idx_future] != z)[0]
            # if len(c) > 0:
            #     print(c)
            #     print(z[c[0]])
            #     print(self._z[idx_future][c[0]])
            #     print(idxs[c[0]])
            #     print(idx_future[c[0]])
            #     print(a[c[0]])
            #     print(b[c[0]])
            #     print(self._step2trajend[idxs[c[0]]])
            #     print(self._truncated[a[c[0]]:b[c[0]]])

            # assert torch.allclose(self._z[idx_future], z)

            output.future = (
                get_obs_by_idx(self._obs, idx_future),
                get_obs_by_idx(self._next_obs, idx_future),
                self._action[idx_future],
            )
            
        return output

    
    @torch.no_grad()
    def sample_start(self, batch_size):
        idx = torch.where(self._timesteps == 0)[0]
        select = torch.from_numpy(np.random.choice(len(idx), batch_size, replace=not self._full)).to(self.device)
        idx = idx[select]
        obs = self._obs[idx] if not isinstance(self._obs, dict) else {k: v[idx.cpu()].to(self.device) for k, v in self._obs.items()}
        return obs, self._z[idx], self._timesteps[idx]

        
    def save(self, path):
        import torch
        torch.save(self, path)