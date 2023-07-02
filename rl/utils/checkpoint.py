import torch
import os
import os.path as osp
import numpy as np
from tools import Configurable
from .evaluator import Evaluator


class ModelCheckpoint(Configurable):
    def __init__(self,
                 path,
                 evaluator: Evaluator, # we save the evaluator here 
                 cfg=None,
                 keep_highest=None,
                 save_latest=False,
                 save_interval=0,
                 monitor_name='score'
                 ):
        Configurable.__init__(self)
        # print(evaluator)
        # print(evaluator.keep_highest)

        if evaluator is not None:
            if hasattr(evaluator, 'keep_highest'):
                keep_highest = evaluator.keep_highest
            monitor_name = evaluator.monitor_name

        if keep_highest is None:
            keep_highest = True

        self.keep_highest = keep_highest

        self.epoch = -1
        self.best_epoch = -1
        self.best_val = -np.inf if self.keep_highest else np.inf

        self.save_latest = save_latest
        self.save_interval = save_interval

        self.path = osp.join(path, f"checkpoints")
        os.makedirs(self.path, exist_ok=True)

        self.evaluator = evaluator
        self.monitor_name = monitor_name


    def log(self, model, epoch=None, eval_status: dict = None):
        def save_model(state_dict, eval_status, save_path):
            if self.evaluator is not None:
                self.evaluator.dump(save_path, eval_status)

            print('saving to ..', save_path)
            torch.save(state_dict, save_path + '.ckpt')

        self.epoch = epoch or self.epoch + 1

        if eval_status is None and self.evaluator is not None:
            eval_status = self.evaluator(model)

        if eval_status is None:
            score = None
        else:
            eval_status['epoch'] = self.epoch
            score = eval_status.get(self.monitor_name)

        print('log to', self.path)
        print('monitor name', self.monitor_name, 'score:', score)

        state_dict = None
        if score is not None:
            if (self.keep_highest and score > self.best_val) or (not self.keep_highest and score < self.best_val):

                self.best_val = score
                save_model(
                    state_dict or model.state_dict(),
                    eval_status,
                    osp.join(self.path, 'best'))

        if self.save_latest:
            save_model(
                state_dict or model.state_dict(),
                eval_status,
                osp.join(self.path, 'latest'))

        if self.save_interval and self.epoch % self.save_interval:
            save_model(
                state_dict or model.state_dict(),
                eval_status,
                osp.join(self.path, f'epoch-{self.epoch}'))