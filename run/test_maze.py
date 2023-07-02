import tqdm
import numpy as np
from rpg.env_base import GymVecEnv, TorchEnv
from rpg.soft_rpg import Trainer
from rpg.maze_tester.maze_exp import base_config

if __name__ == '__main__':
    N = 100
    
    #env = TorchEnv('SmallMaze', N, ignore_truncated_done=True, reward=False)
    # 150000

    trainer = Trainer.parse(
        None, 
        **base_config
    ) # do not know if we need max_grad_norm
    # breakpoint()
    trainer.run_rpgm()