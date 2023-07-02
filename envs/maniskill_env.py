import gym

def make(**kwargs):
    from envs.maniskill.open_carbinet import OpenCabinetDoorEnv
    pid = '1018'
    return OpenCabinetDoorEnv(**{'reward_mode': 'dense', 'obs_mode': 'state', 'model_ids': [pid], 'fixed_target_link_idx': 1}, **kwargs)

if __name__ == '__main__':
    env = make()
    images = []
    env.reset()
    images.append(env.render(mode='rgb_array'))
    for i in range(100):
        env.step(env.action_space.sample())
        images.append(env.render(mode='rgb_array'))
    from tools.utils import animate
    animate(images, 'output.mp4')