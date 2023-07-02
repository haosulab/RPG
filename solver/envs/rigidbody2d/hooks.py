import os
import code
import numpy as np
from tools.utils import logger
from solver.train_rpg import Trainer, GoalEnv
import torch

@torch.no_grad()
def record_rollout(trainer: Trainer, mainloop_locals, batch_size=None, save_last=True):

    """ hook 1: render trajectory sample(s) """

    print("----- recording batch/single rollout video -----")
    print(f"{trainer.rpg._cfg.weight.reward = }")

    epoch_id = mainloop_locals["epoch_id"]
    if epoch_id % trainer._cfg.record_gif_per_epoch == 0:
        trainer.env.eval()

        if batch_size is not None:
            tmp_batch_size = trainer.env.n_batches
            trainer.env.n_batches = batch_size
        os.makedirs(os.path.join(logger.get_dir(), "rollout"), exist_ok=True)

        traj = trainer.rpg.inference(trainer.env)
        imgs = [info_dict["img"] for info_dict in traj['info']]
        logger.animate(imgs, f"rollout/{epoch_id:03d}_batch.mp4")
        if save_last:
            os.system(
                f"cp {logger.get_dir()}/rollout/{epoch_id:03d}_batch.mp4 {logger.get_dir()}/rollout/last_batch.mp4"
            )
            

        env_single_batch_cfg = trainer._cfg.env
        env_single_batch_cfg["n_batches"] = 1
        env_single_batch = GoalEnv.build(env_single_batch_cfg)
        env_single_batch.eval()

        traj = trainer.rpg.inference(env_single_batch)
        imgs = [info_dict["img"] for info_dict in traj['info']]
        logger.animate(imgs, f"rollout/{epoch_id:03d}_single.mp4")
        if save_last:
            os.system(
                f"cp {logger.get_dir()}/rollout/{epoch_id:03d}_single.mp4 {logger.get_dir()}/rollout/last_single.mp4"
            )
        trainer.env.train()

        if batch_size is not None:
            trainer.env.n_batches = tmp_batch_size


@torch.no_grad()
def save_traj(trainer: Trainer, mainloop_locals):
    print("----- saving trajs -----")
    epoch_id = mainloop_locals["epoch_id"]
    if trainer._cfg.save_traj.per_epoch > 0 and epoch_id % trainer._cfg.save_traj.per_epoch == 0:
        os.makedirs(os.path.join(logger.get_dir(), f"trajs"), exist_ok=True)
        
        trajs = []
        for _ in range(trainer._cfg.save_traj.num_iter):
            traj = trainer.rpg.inference(trainer.env, return_state=True)
            trajs.append(traj)

        logger.torch_save(trajs, f"trajs/{epoch_id}")
        print(f"save_traj: traj saved to {logger.get_dir()}/trajs/{epoch_id}")


def print_gpu_usage(trainer, mainloop_locals):

    """ hook 2: render trajectory sample(s) """

    MB = 1024 ** 2
    total_memory = torch.cuda.get_device_properties(0).total_memory / MB
    reserved = torch.cuda.memory_reserved(0) / MB
    allocated = torch.cuda.memory_allocated(0) / MB
    free_memory = reserved - allocated  # free inside reserved
    print("--------------------------")
    print(f"total     mem: {total_memory:.2f}")
    print(f"reserved  mem: {reserved    :.2f}")
    print(f"allocated mem: {allocated   :.2f}")
    print(f"free      mem: {free_memory :.2f}")
    print("--------------------------")

