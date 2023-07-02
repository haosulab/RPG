"""Environments using kitchen and Franka robot."""
import os
import numpy as np
# 'microwave', 'kettle', 'light switch', 'slide cabinet'
os.environ['MUJOCO_GL'] = 'egl'
import d4rl # must import ...
import gym
from dm_control.mujoco import engine
from d4rl.kitchen.adept_envs.utils.configurable import configurable
from d4rl.kitchen.adept_envs.franka.kitchen_multitask_v0 import KitchenTaskRelaxV1

from d4rl.offline_env import OfflineEnv

OBS_ELEMENT_INDICES = {
    'bottom burner': np.array([11, 12]),
    'top burner': np.array([15, 16]),
    'light switch': np.array([17, 18]),
    'slide cabinet': np.array([19]),
    'hinge cabinet': np.array([20, 21]),
    'microwave': np.array([22]),
    'kettle': np.array([23, 24, 25]),#, 26, 27, 28, 29]),
}
OBS_ELEMENT_GOALS = {
    'bottom burner': np.array([-0.88, -0.01]),
    'top burner': np.array([-0.92, -0.01]),
    'light switch': np.array([-0.69, -0.05]),
    'slide cabinet': np.array([0.37]),
    'hinge cabinet': np.array([0., 1.45]),
    'microwave': np.array([-0.75]),
    'kettle': np.array([-0.23, 0.75, 1.62]),#, 0.99, 0., 0., -0.06]),
    }
# BONUS_THRESH = 0.3

@configurable(pickleable=True)
class KitchenBase(KitchenTaskRelaxV1, OfflineEnv):
    # A string of element names. The robot's task is then to modify each of
    # these elements appropriately.
    TASK_ELEMENTS = []
    REMOVE_TASKS_WHEN_COMPLETE = False
    TERMINATE_ON_TASK_COMPLETE = False
    partial_reward=True

    BONUS_THRESH = 0.3 

    def __init__(self, dataset_url=None, n_block=4, reward_type='sparse', obs_dim=6, ref_max_score=None, ref_min_score=None, **kwargs):
        self.TASK_ELEMENTS = self.TASK_ELEMENTS[:n_block]
        self.reward_type = reward_type
        self.embedder = None
        self.tasks_to_complete = set(self.TASK_ELEMENTS)
        self.obs_dim = obs_dim
        super(KitchenBase, self).__init__(**kwargs)
        OfflineEnv.__init__(
            self,
            dataset_url=dataset_url,
            ref_max_score=ref_max_score,
            ref_min_score=ref_min_score)

        self.observation_space = gym.spaces.Box(-1., 1., shape=self.reset().shape)

    def _get_obs(self):
        obs = super(KitchenBase, self)._get_obs()

        next_q_obs = self.obs_dict['qp']
        next_obj_obs = self.obs_dict['obj_qp']
        next_goal = self.obs_dict['goal']
        idx_offset = len(next_q_obs)
        idx_offset = len(next_q_obs)


        from envs.utils import get_embeder_np, symlog
        if self.embedder is None:
            self.embedder, _ = get_embeder_np(self.obs_dim, 1)

        outs = []
        dofs = 0
        for element in self.tasks_to_complete:
            element_idx = OBS_ELEMENT_INDICES[element]
            inp = next_obj_obs[..., element_idx - idx_offset]
            outs.append(self.embedder(symlog(inp))/len(inp))
            dofs += len(inp)
        self.dofs = dofs
        ee_id = self.model.site_name2id("end_effector")
        ee_pos = self.data.site_xpos[ee_id]

        outs.append(self.embedder(ee_pos - np.array([-0.4, 0., 1.2])))
        inp = np.concatenate(outs, axis=-1)

        obs = np.concatenate([inp * 0.05, obs * 0.05, inp], axis=-1)
        return obs

    def _get_task_goal(self):
        new_goal = np.zeros_like(self.goal)
        for element in self.TASK_ELEMENTS:
            element_idx = OBS_ELEMENT_INDICES[element]
            element_goal = OBS_ELEMENT_GOALS[element]
            new_goal[element_idx] = element_goal

        return new_goal

    def reset_model(self):
        self.tasks_to_complete = set(self.TASK_ELEMENTS)
        return super(KitchenBase, self).reset_model()

    def _get_reward_n_score(self, obs_dict):
        reward_dict, score = super(KitchenBase, self)._get_reward_n_score(obs_dict)
        reward = 0.
        next_q_obs = obs_dict['qp']
        next_obj_obs = obs_dict['obj_qp']
        next_goal = obs_dict['goal']
        idx_offset = len(next_q_obs)
        completions = []
        reward_dict['metric'] = {}
        for element in self.tasks_to_complete:
            element_idx = OBS_ELEMENT_INDICES[element]
            distance = np.linalg.norm(
                next_obj_obs[..., element_idx - idx_offset] -
                next_goal[element_idx])
            
            complete = distance < self.BONUS_THRESH
            if complete:
                completions.append(element)

            reward_dict['metric'][element.replace(' ', '_')] = complete

        if self.REMOVE_TASKS_WHEN_COMPLETE:
            [self.tasks_to_complete.remove(element) for element in completions]
        bonus = float(len(completions))
        reward_dict['bonus'] = bonus
        reward_dict['r_total'] = len(completions) == len(self.tasks_to_complete)
        reward_dict['metric']['true'] = (len(completions) == len(self.tasks_to_complete))

        if self.reward_type == 'sparse':
            if self.partial_reward:
                reward_dict['r_total'] = bonus / len(self.tasks_to_complete) + (len(completions) == len(self.tasks_to_complete))
            else:
                reward_dict['r_total'] = len(self.tasks_to_complete) == len(completions)
            

        elif self.reward_type == 'bonus':
            #     reward_dict['r_total'] += 0.2 * bonus / 5.
            # elif self.reward_type == 'bonus2':
            #     reward_dict['r_total'] = bonus / len(self.tasks_to_complete)
            # else:
            raise NotImplementedError(self.reward_type)
        else:
            raise NotImplementedError
        
        score = bonus
        return reward_dict, score

    def step(self, a, b=None):
        obs, reward, done, env_info = self._step(a, b=b)
        return obs, reward, done, env_info


    def render(self, mode='rgb_array'):
        camera = engine.MovableCamera(self.sim, 224, 224)
        camera.set_pose(distance=2.2, lookat=[-0.2, .5, 2.], azimuth=70, elevation=-35)
        img = camera.render()
        return img

    def _step(self, a, b=None):
        a = np.clip(a, -1.0, 1.0)

        if not self.initializing:
            a = self.act_mid + a * self.act_amp  # mean center and scale
        else:
            self.goal = self._get_task_goal()  # update goal if init
            

        self.robot.step(
            self, a, step_duration=self.skip * self.model.opt.timestep)

        # observations
        obs = self._get_obs()

        #rewards
        reward_dict, score = self._get_reward_n_score(self.obs_dict)

        # termination
        done = False

        # finalize step
        env_info = {
            'time': self.obs_dict['t'],
            'obs_dict': self.obs_dict,
            'rewards': reward_dict,
            'success': score,
            'metric': reward_dict['metric'],
            # 'images': np.asarray(self.render(mode='rgb_array'))
        }
        # self.render()
        return obs, reward_dict['r_total'], done, env_info

    def _render_traj_rgb(self, traj, occ_val=False, history=None, verbose=True, **kwargs):
        # don't count occupancy now ..
        from .. import utils
        obs = utils.extract_obs_from_tarj(traj) / 0.05
        obs = obs[..., :self.dofs]
        outs = dict(occ=utils.count_occupancy(obs, -1., 1., n_bin=5))
        history = utils.update_occupancy_with_history(outs, history)
        output = {
            'background': {},
            'history': history,
            'image': {},
            'metric': {k: (v > 0.).mean() for k, v in history.items()},
        }
        return output


class KitchenMicrowaveKettleLightSliderV0(KitchenBase):
    TASK_ELEMENTS = ['microwave', 'kettle', 'light switch', 'slide cabinet']



class KitchenV2(KitchenBase):
    TASK_ELEMENTS = ['microwave', 'slide cabinet']
    partial_reward=False


class KitchenV3(KitchenBase):
    TASK_ELEMENTS = ['microwave', 'slide cabinet', 'hinge cabinet']
    partial_reward=False

class KitchenV4(KitchenBase):
    TASK_ELEMENTS = ['microwave', 'slide cabinet']
    partial_reward=False
    BONUS_THRESH = 0.1


if __name__ == '__main__':
    env = KitchenV4(n_block=1)
    env.reset()
    images = []
    for i in range(1000):
        _, r, done, info = env.step(env.action_space.sample())
        
        images.append(env.render())
        
        assert not done
        if (i + 1) % 280 == 0:
            print(i)
            env.reset()
    from tools.utils import animate
    animate(images)