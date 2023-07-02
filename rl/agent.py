# batch RL agents.
import torch
import numpy as np
from .optims import Optim
from .models import Actor, PPOHead

from tools import Configurable, as_builder, dist_utils
from tools.utils import RunningMeanStd, logger

from .utils import RLEvaluator, ModelCheckpoint
from torch import nn
from tools.dist_utils import get_rank
from tools.nn_base import Network


class Agent(Network):
    def __init__(
        self,
        obs_space,
        action_space,
        cfg=None,
        auxiliary=None,
        actor=Actor.get_default_config(),

        obs_norm=True,
        reward_norm=True,
        obs_norm_clip=10.,

        # for eval and checkpoint
        eval_episode=100,  # by default, evaluate per 100 epoch..
        evaluator_cfg=RLEvaluator.get_default_config(),
        model_checkpoint_cfg=ModelCheckpoint.get_default_config(),

        ignore_episode_done=True,
        save_episode=None,
    ):
        Network.__init__(self)

        self.obs_space = obs_space
        self.action_space = action_space

        self.actor = self.build_actor(obs_space, action_space, actor)


        self._obs_rms = RunningMeanStd(
            clip_max=obs_norm_clip) if obs_norm else None
        self._rew_rms = RunningMeanStd(last_dim=False) if reward_norm else None

        #  config evaluator ..
        assert eval_episode is None or save_episode is None, "Either eval or save or save without eval .."
        self.save_episode = eval_episode or save_episode
        if self.save_episode:
            print(f'will save model per {self.save_episode} epoch ..')
            self.evaluator = RLEvaluator(None, cfg=evaluator_cfg)  # set None
            self.model_saver = ModelCheckpoint(
                logger.get_dir(), self.evaluator, model_checkpoint_cfg
            )

    def build_actor(self, obs_space, action_space, actor_cfg):
        return Actor(obs_space, action_space, self._cfg.auxiliary, cfg=actor_cfg)

    def to(self, device):
        self.device = device
        return nn.Module.to(self, device)


    def process_obs(self, x: np.ndarray):
        if self._obs_rms is not None:
            if self.training:
                self._obs_rms.update(x)
            # interesting, this works even if x is a list
            x = self._obs_rms.normalize(x)
        return x

    def select_action(self, obs):
        # note that this is original env observation
        with torch.no_grad():
            return self.actor(self.process_obs([obs])).mean[0].detach().cpu().numpy()


    def state_dict(self):
        state = super().state_dict()
        state['_obs_rms'] = self._obs_rms
        state['_rew_rms'] = self._rew_rms
        return state

    def load_state_dict(self, state_dict):
        self._obs_rms = state_dict.pop('_obs_rms')
        self._rew_rms = state_dict.pop('_rew_rms')
        super().load_state_dict(state_dict)
        return dict

    
    def eval_and_save(self, envs, epoch=None, **kwargs):
        if get_rank() == 0:
            if hasattr(self, 'train_envs'):
                self.rolling_state = None  # clear the rolling state ..
            if self.evaluator is not None:
                self.evaluator.envs = envs
            #logger.log_model(self, **kwargs)
            self.model_saver.log(self, epoch=epoch, **kwargs)