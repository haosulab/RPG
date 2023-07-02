import tqdm
import os
import torch
from torch import nn
from .config import Configurable, merge_inputs
from .nn_base import Network
from .utils import info_str, logger, summarize_info
import numpy as np
from collections import deque
from tools.utils import logger
from . import dist_utils

class BatchGen:
    # hack now ..
    def __init__(self, buffer, batch_size, iters, KEYS):
        self.buffer = buffer
        self.batch_size = batch_size
        self.iters = iters
        self.KEYS = KEYS

    def __iter__(self):
        for _ in range(self.iters):
            index = np.random.choice(len(self.buffer), self.batch_size)
            data = [self.buffer[i] for i in index]
            yield {
                k: [i[idx] for i in data] for idx, k in enumerate(self.KEYS)
            }

    def __len__(self):
        return self.iters


class OptimModule(Network):
    KEYS = []
    name = None
    def __init__(
        # maxlen is the length of the buffer
        self, network, cfg=None,
        lr=3e-4, max_grad_norm=None, eps=1e-8, loss_weight=None, verbose=True,
        training_iter=1, batch_size=128, maxlen=3000,
        accumulate_grad=0, mode='not_step',
    ):
        super().__init__()
        self.network = network
        dist_utils.sync_networks(self.network)
        #for idx, i in enumerate(self.network.parameters()):
        #    print(idx, i.data.sum())
        self.params = list(network.parameters() if isinstance(network, nn.Module) else [network])
        self.optimizer = torch.optim.Adam(self.params, lr=cfg.lr, eps=cfg.eps)
        self.buffer = deque(maxlen=maxlen)

        if accumulate_grad > 1:
            self.n_steps = 0

    @property
    def active(self):
        return self._cfg.training_iter

    def __call__(self, obs, **kwargs):
        return self.forward_network(obs, **kwargs)

    def forward_network(self, obs, **kwargs):
        return self.network(obs, **kwargs)

    def optim_step(self):
        dist_utils.sync_grads(self.network)
        if self._cfg.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.params, self._cfg.max_grad_norm)
        self.optimizer.step()

    def compute_loss(self, data, label):
        # mse loss by default ..
        out = self.forward_network(data)
        assert out.shape == label.shape, f"{out.shape} {label.shape}"
        return torch.nn.functional.mse_loss(out, label)

    def weight_loss(self, loss, loss_weight=None):
        if loss_weight is None:
            loss_weight = self._cfg.loss_weight
        if isinstance(loss, dict):
            info = loss
            if 'loss' not in info:
                assert self._cfg.loss_weight is not None
                total = 0.
                for i in loss:
                    weight = float(loss_weight[i])
                    total = total + loss[i] * weight
                loss = total
                info['loss'] = loss.item()
                raise NotImplementedError("Never checked again")
            else:
                loss = info['loss']
                info['loss'] = loss.item()
        else:
            if loss_weight is None:
                loss_weight = 1.
            assert isinstance(loss_weight, float)
            info=dict(loss=loss.item())
            loss = loss * loss_weight
        return loss, info

    def _compute_loss(self, *args, backward=True, **data):
        # compute loss and do backward if necessary
        loss = self.compute_loss(*args, **data)
        loss, info = self.weight_loss(loss)
        if backward:
            loss.backward()
        return loss, info

    def step(self, *args, backward=True, **data):
        assert self._cfg.mode == 'step', "please set the mode to be step to use this mode.. as we do not support accumulate_grad in this mode"
        # one step optimization
        if backward:
            self.optimizer.zero_grad()
        loss, info = self._compute_loss(*args, backward=backward, **data)
        if backward:
            self.optim_step()
        return loss, info


    def train_loop(self, batches, logger=None, mode='train', max_iter=None, loss_kwargs={}, return_info=False, **kwargs):
        self.train()
        cfg = merge_inputs(self._cfg, **kwargs)
        if cfg.verbose:
            batches = tqdm.tqdm(batches, total=len(batches))
        
        prefix = self.name + '_' if self.name is not None else ''

        backward = (mode == 'train')
        if return_info:
            infos = []
        for idx, data in enumerate(batches):
            if max_iter and idx >= max_iter: # max_iter should not be None
                break
            _, info = self.step(backward=backward, **data, **loss_kwargs)
            if return_info:
                infos.append(info)
            if cfg.verbose:
                batches.set_description(
                    info_str(idx, **info)
                )
            if logger is not None:
                logger.logkvs_mean({
                    mode + f'/{prefix}{k}': v for k, v in info.items()
                })
        if return_info:
            return summarize_info(infos, reduce='mean')


    def clear_buffer(self):
        # self.buffer.clear() 
        raise NotImplementedError

    def append_data(self, **kwargs):
        self.buffer.append([kwargs[i] for i in self.KEYS])

    def optim(self, trajs, logger, mode, **kwargs):
        for i in trajs:
            self.append_data(**i)
        self.train_loop(self.datagen(), logger, mode, **kwargs)

    def datagen(self):
        return BatchGen(self.buffer, self._cfg.batch_size, self._cfg.training_iter, self.KEYS)

    def resume(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu'))
    
    def save(self, val):
        if self.active:
            assert self.name is not None
            logger.torch_save(self.state_dict(), self.name, val)

    def optimize(self, loss: torch.Tensor):
        if self._cfg.accumulate_grad > 1:
            if self.n_steps % self._cfg.accumulate_grad == 0:
                self.zero_grad()

            (loss / self._cfg.accumulate_grad).backward()
            if (self.n_steps+1) % self._cfg.accumulate_grad == 0:
                self.optim_step()

            self.n_steps += 1
        else:
            self.zero_grad()
            loss.backward()
            self.optim_step()
            
class LossOptimizer(OptimModule):
    def __init__(self, network, cfg=None):
        super().__init__(network)
            
            
class TrainerBase(Configurable):
    def __init__(self, cfg=None,
        path=None,
        format_strs='stdout+csv',
        seed=None,
        MODEL_PATH=None,
        log_date=False,
    ):
        super().__init__()
        self.MODEL_PATH = MODEL_PATH

        config_path = logger.configure(self.get_path(path), format_strs=format_strs.split('+'), date=log_date, config=cfg)
        print('config path..', config_path)
        with open(os.path.join(config_path, 'config.yml'), 'w') as f:
            f.write(str(self._cfg))

        if seed is not None:
            from .utils import seed as seedit
            seedit(seed)

    def get_path(self, path):
        if path is None or self.MODEL_PATH is None:
            return path
        if self.MODEL_PATH is not None:
            return os.path.join(self.MODEL_PATH, path)

    def prepare(self):
        resume_path = self.get_path(self._cfg.resume_path)
        if resume_path is not None:
            print(f'Resume {resume_path}!')
            self.resume(resume_path)

    @property
    def modules(self):
        raise NotImplementedError

    def resume(self, pretrained):
        if pretrained is not None:
            for module in self.modules:
                module.resume(f'{pretrained}/{module.name}')

    def train(self):
        for module in self.modules:
            module.train()

    def eval(self):
        for module in self.modules:
            module.eval()

    def optim_all(self, data):
        for module in self.modules:
            module.optim(data, logger, 'train')



    @classmethod
    def resume_from_exp_path(cls, exp_path, target_path=None):
        import yaml
        from tools import CN
        with open(os.path.join(exp_path, 'config.yml'), 'r') as f:
            config = CN(yaml.safe_load(f))
        trainer = cls(cfg=config, path=target_path)
        trainer.resume(exp_path)
        return trainer