# simple hacks to build various envs ..
import gym
import numpy as np


def make_move():
    from mpm.env import DifferentiablePhysicsEnv

    env = DifferentiablePhysicsEnv(
        observer_cfg={"TYPE": "ParticleObserver"},
        max_episode_steps=50
    )

    state = env.get_state()

    def loss_fn(idx, **kwargs):
        return kwargs['dist'].min(axis=0)[0].sum()

    env.set_loss_fn(loss_fn)
    env.set_initial_state_sampler(lambda env: state)

    return env


class ManiSkillWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = env.observation_space
        self.size = self.observation_space['pointcloud']['xyz'].shape[0]
        self._rl_steps = 0

    def _wrap_obs(self, obs):
        idx = np.random.choice(self.size, 1000)

        return {
            'agent': obs['agent'],
            'pointcloud': {k: i[idx] for k, i in obs['pointcloud'].items()},
        }

    def reset(self):
        self._rl_steps = 0
        return self._wrap_obs(self.env.reset())

    def step(self, action):
        self._rl_steps += 1
        obs, r, done, info = self.env.step(action)
        done = (self._rl_steps >= 200)  # 200 steps
        if done:
            info['TimeLimit.truncated'] = True
        return self._wrap_obs(obs), r, done, info

    def render(self, mode='rgb_array'):
        return np.uint8(self.env.render('color_image')['world']['rgb'] * 255)


def make_carbinet():
    import gym
    import mani_skill.env
    env = gym.make('OpenCabinetDrawer_1000_link_0-v0')
    env.set_env_mode(obs_mode='pointcloud', reward_type='dense')
    return ManiSkillWrapper(env)
