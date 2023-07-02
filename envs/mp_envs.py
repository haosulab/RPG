# make motion planning environments

def make(env_name):
    if env_name == 'LargeMaze':
        #from envs.maze import LargeMaze
        from rpg.env_base import TorchEnv
        return TorchEnv('LargeMaze', 128)

    else:
        raise NotImplementedError