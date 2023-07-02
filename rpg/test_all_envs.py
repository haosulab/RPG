from env_base import GymVecEnv
from gym.wrappers import TimeLimit
import gym


def make_env(env_name):
    from gym.wrappers import TimeLimit

    if env_name == 'BlockPush2':
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        from envs.block import BlockEnv
        return TimeLimit(BlockEnv(n_block=2), 60)

    elif env_name == 'CabinetDense':
        from envs.maniskill_env import make
        print(f"this is dense Cabinet")
        return TimeLimit(make(obs_dim=0, reward_type="dense"), 60)

    elif env_name == 'EEArm':
        from envs.maniskill.ee_arm import EEArm
        return TimeLimit(EEArm(obs_dim=6, reward_mode="sparse"), 60)

    elif env_name == 'AdroitHammer':
        from envs.modem.adroit import make_adroit_env
        return TimeLimit(make_adroit_env('hammer-v0'), 125)

    elif env_name == 'AdroitDoor':
        from envs.modem.adroit import make_adroit_env
        return TimeLimit(make_adroit_env('door-v0'), 100)

    elif env_name == 'MWStickPull':
        from envs.modem.metaworld_envs import make_metaworld_env
        return TimeLimit(make_metaworld_env('stick-pull'), 100)

    elif env_name == 'MWBasketBall':
        from envs.modem.metaworld_envs import make_metaworld_env
        return TimeLimit(make_metaworld_env('basketball'), 100)

    elif env_name == 'AntPushDense':
        from envs.ant_envs import AntHEnv
        print(f"this is dense antpush")
        return TimeLimit(AntHEnv('AntPush', obs_dim=0, reward_type="dense"), 400)
    
    return gym.make(env_name)


if __name__ == "__main__":

    envs = [
        "AdroitHammer",
        "AdroitDoor",
        "AntPushDense",
        "BlockPush2", 
        "CabinetDense", 
        "EEArm",
        "MWStickPull",
        "MWBasketBall",
    ]

    for env_name in envs:
        # env = GymVecEnv(env_name, n=1)
        # obs, timestep = env.start()
        # print(env_name, obs.shape)
        env = make_env(env_name)
        print(env_name, "OK")