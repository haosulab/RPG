import os
from tools.utils import logger
from solver.train_rpg import Trainer, GoalEnv
import torch

@torch.no_grad()
def record_rollout(trainer: Trainer, mainloop_locals):

    """ hook 1: render trajectory sample(s) """

    print("----- recording batch/single rollout video -----")
    print(f"{trainer.rpg._cfg.weight.reward = }")

    epoch_id = mainloop_locals["epoch_id"]
    if epoch_id % trainer._cfg.record_gif_per_epoch == 0:
        trainer.env.eval()
        os.makedirs(os.path.join(logger.get_dir(), "rollout"), exist_ok=True)

        traj = trainer.rpg.inference(trainer.env)
        imgs = [info_dict["img"] for info_dict in traj['info']]
        video_name = f"{epoch_id:03d}_batch.mp4"

        logger.animate(imgs, f"rollout/{video_name}")
        os.system(f"cp {logger.get_dir()}/rollout/{video_name} {logger.get_dir()}/rollout/last.mp4")

        trainer.env.train()


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

