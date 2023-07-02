import torch
import numpy as np
from tools.config import Configurable


## add support for dict later ..
class ReplayBuffer(Configurable):
    def __init__(self, state_space, action_space,
                 cfg=None, memory_size=int(1e6), batch_size=256):
        super(ReplayBuffer, self).__init__(cfg)

        self.state_shape = state_space.shape
        self.action_shape = action_space.shape
        self.batch_size = batch_size

        assert isinstance(self.state_shape, tuple)
        assert isinstance(self.action_shape, tuple)

        self._n = 0
        self._p = 0

        self.memory_size = memory_size
        assert isinstance(memory_size, int) and memory_size > 0

        self.states = np.empty(
            (self.memory_size,) + self.state_shape, dtype=np.float32)
        self.next_states = np.empty(
            (self.memory_size,) + self.state_shape, dtype=np.float32)
        self.actions = np.empty(
            (self.memory_size,) + self.action_shape, dtype=np.float32)

        self.rewards = np.empty((self.memory_size, 1), dtype=np.float32)
        self.dones = np.empty((self.memory_size, 1), dtype=np.float32)
        self.batch_size = batch_size

    def to_tensor(self, x, dtype=torch.float):
        return torch.tensor(
            x, dtype=dtype, device=self.device
        )

    def append(self, state, action, reward, next_state, done, real_done, **kwargs):
        self.states[self._p, ...] = state
        self.actions[self._p, ...] = action
        self.rewards[self._p, ...] = reward
        self.next_states[self._p, ...] = next_state
        self.dones[self._p, ...] = done

        # for i, v in kwargs.items():
        #    self.__getattribute__(i)[self._p, ...] = v
        self._n = min(self._n + 1, self.memory_size)
        self._p = (self._p + 1) % self.memory_size

    def select(self, ind):
        batch = {
            'state': self.states[ind],
            'action': self.actions[ind],
            'reward': self.rewards[ind],
            'next_state': self.next_states[ind],
            'done': self.dones[ind]
        }
        return batch

    def sample(self):
        ind = np.random.randint(low=0, high=self._n, size=self.batch_size)
        return self.select(ind)

    def __len__(self):
        return self._n
