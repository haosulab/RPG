import os
import sys
import torch
import tqdm
import matplotlib.pyplot as plt

from tools.optim import TrainerBase
from tools.utils import logger, info_str, summarize_info, dconcat, dslice, dshape
from gym.spaces import Box, Discrete

from .envs import GoalEnv
from .actor import Backbone
from .rpg import RPG
from .rpg_net import RPGActor, RPGCritic, InfoNet, PriorActor
from .utils import create_z_space


class Trainer(TrainerBase):
    def __init__(
        self, cfg=None,

        env=GoalEnv.to_build(TYPE="Bandit"),
        rpg=RPG.get_default_config(),

        z_dim=5, z_cont_dim=0, K=0,

        actor=RPGActor.to_build(TYPE=RPGActor),
        info_net=InfoNet.to_build(TYPE=InfoNet),
        
        backbone=Backbone.to_build(TYPE='MLP'),

        n_batches=100,
        batch_size=128,
        max_epochs=100,

        record_gif=True,

        increasing_reward=0,
        record_gif_per_epoch=0,

        save_traj=dict(
            per_epoch=1,
            num_iter=1
        ),
        check_dir=False,

        format_strs='tensorboard+csv+stdout',
        return_state=False,

        device = 'cuda:0',
        mutual_decay=1.,
        ent_z_decay = 1.,
        start_pg_epoch=None,


        softmax = dict(
            decay=0.,
            initial=10000,
            decay_epoch=20
        )
    ):
        if check_dir and os.path.exists(self._cfg.path):
            exit(0)
        super().__init__()
        z_space = create_z_space(z_dim, z_cont_dim)

        self.env: GoalEnv = GoalEnv.build(env)

        obs_space = self.env.observation_space
        action_space = self.env.action_space

        actor = RPGActor.build(
            obs_space, action_space, z_space, K,
            cfg=actor, backbone=backbone, env_low_steps=self.env.low_steps).to(device)
        prior_actor = PriorActor(
            obs_space, action_space, z_space, K).to(device)
        critic = RPGCritic(
            obs_space, z_space, K, backbone=backbone).to(device)
        info_log_q = InfoNet.build(
            obs_space, action_space, z_space, backbone=backbone, cfg=info_net).to(device)

        self.rpg = RPG(actor, info_log_q, critic, prior_actor, cfg=rpg)
        self.epoch_hooks = []


        if K > 0:
            logger.log(f"K={K}, expected {(self.env.low_steps * self.env.num_stages+1)//K} options in total.")
        
        # record terminal start command
        with open(f"{logger.get_dir()}/cmdargs.sh", "w") as f:
            f.write(" ".join(sys.argv))

    @torch.no_grad()
    def visualize_traj(self, info=None):
        traj = self.rpg.inference(
            self.env, 
            batch_size=self._cfg.batch_size*10, return_state=self._cfg.return_state)

        # render states ..
        last = dslice(traj['last_obs'], (None, slice(None, None, None)))
        img = self.env._render_traj_rgb(
            dconcat(traj['s'], last), traj=traj, info=info) 
        return img

    def resume(self, exp_path):
        import os
        self.rpg = torch.load(os.path.join(exp_path, 'model'))

    def start(self, max_epochs=None):
        max_epochs = max_epochs or self._cfg.max_epochs

        epoch_id = 0

        gifs = []
        reward_weight = self.rpg._cfg.weight.reward
        while True:

            if self._cfg.increasing_reward:
                self.rpg._cfg.weight.reward = reward_weight * ((epoch_id+1)/max_epochs)
                print(f"increasing reward weight to {self.rpg._cfg.weight.reward}")

            if self._cfg.mutual_decay and epoch_id > 0:
                self.rpg._cfg.weight.mutual_info = self.rpg._cfg.weight.mutual_info * self._cfg.mutual_decay
                print(f"decaying mutual info to {self.rpg._cfg.weight.mutual_info}")

            if self._cfg.ent_z_decay and epoch_id > 0:
                self.rpg._cfg.weight.ent_z = self.rpg._cfg.weight.ent_z * self._cfg.ent_z_decay
                print(f"decaying ent_z to {self.rpg._cfg.weight.ent_z}")

            if self._cfg.start_pg_epoch is not None and epoch_id > self._cfg.start_pg_epoch:
                print('start pg')
                self.rpg._cfg.stop_pg = False

            if self.rpg.actor._cfg.softmax_policy:
                if epoch_id == 0:
                    self.rpg.actor.softmax_temperature = self._cfg.softmax.initial
                elif self._cfg.softmax.decay_epoch is not None and epoch_id >= self._cfg.softmax.decay_epoch:
                    self.rpg.actor.softmax_temperature *= self._cfg.softmax.decay
                print(f"softmax_temperature {self.rpg.actor.softmax_temperature}")
                

            it = tqdm.trange(self._cfg.n_batches)
            it.set_description(f'epoch {epoch_id}')
            for _ in it:
                info = self.rpg.run_batch(
                    self.env, batch_size=self._cfg.batch_size, info_q_iter=1)
                logger.logkvs_mean(info)

            for hook in self.epoch_hooks:
                hook(self, locals())

            if self._cfg.record_gif:
                # default trajectory saver..
                img = self.visualize_traj(
                    dict(epoch_id=epoch_id)
                )

                logger.savefig('traj.png', img)
                gifs.append(img)

            logger.dumpkvs()
            logger.torch_save(self.rpg, 'model')
            epoch_id += 1
            if epoch_id > max_epochs:
                break

            if len(gifs) > 0 and self._cfg.record_gif_per_epoch > 0 and (epoch_id % self._cfg.record_gif_per_epoch == 0):
                logger.animate(gifs, 'video.mp4', fps=10)

        if len(gifs) > 0:
            logger.animate(gifs, 'video.mp4', fps=10)


if __name__ == '__main__':
    Trainer.parse().start()