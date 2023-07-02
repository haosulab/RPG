import numpy as np
from abc import ABC, abstractmethod
from gym.wrappers import TimeLimit
from tools.utils import tonumpy


class VecEnv(ABC):
    def __init__(self) -> None:
        self.nenv = None
        self.observation_space = None
        self.action_space = None
        self.reward_dim = 1

    @abstractmethod
    def start(self, **kwargs):
        # similar to reset, but return obs [batch of observations: nunmy array/dict/tensors] and timesteps [numpy].
        # if done or no obs, reset it.
        # note that this process is never differentiable.
        pass

    @abstractmethod
    def step(self, action):
        # return {'next_obs': _, 'obs': new_obs, 'reward': reward, 'done': done, 'info': info, 'total_reward': 0., 'truncated': False, 'episode': [], 'timestep'}
        # next obs will not be obs if done.
        pass

    @abstractmethod
    def render(self, mode):
        pass


class GymVecEnv(VecEnv):
    def __init__(self, env_name, n, ignore_truncated_done=True, **kwargs) -> None:
        # by default, the mujoco's gym env do not have a truncated reward. 
        super().__init__()
        self.env_name = env_name

        import gym
        from rl.vec_envs import SubprocVectorEnv
        # if env_name.startswith('Ant') or env_name.startswith('Block') or env_name.startswith('FixArm') or 'Arm' in env_name or 'Adroit' in env_name:
        import multiprocessing as mp
        mp.set_start_method('spawn', force=True)

        def make_env():
            from gym.wrappers import TimeLimit
            # if env_name == 'AntPush':
            #     from envs.ant_envs import AntHEnv
            #     return TimeLimit(AntHEnv(env_name, **kwargs), 60)

            if env_name == 'TripleAnt':
                from envs.triple_ant import TripleAntEnv
                return TimeLimit(TripleAntEnv( **kwargs), 100)

            # elif env_name == 'AntMaze':
            #     from envs.ant_maze import AntMaze
            #     return TimeLimit(AntMaze( **kwargs), 150)

            elif env_name == 'AntMaze2':
                from envs.ant_maze import AntMaze
                return TimeLimit(AntMaze(**kwargs, maze_id=4, init_pos=(1, 1)), 150)


            elif env_name == 'AntMaze3':
                from envs.ant_maze import AntCross
                return TimeLimit(AntCross(**kwargs), 200)

            elif env_name == 'AntFork':
                from envs.ant_maze import AntFork
                return TimeLimit(AntFork(**kwargs), 200)

            elif env_name == 'BlockPush':
                import os
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                from envs.block import BlockEnv
                return TimeLimit(BlockEnv(**kwargs), 100)

            elif env_name == 'BlockPush2':
                import os
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                from envs.block import BlockEnv
                return TimeLimit(BlockEnv(**kwargs, n_block=2), 60)


            elif env_name == 'Fetch':
                # import os
                # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                from envs.fetch.sequential import SequentialStack
                return TimeLimit(SequentialStack(**kwargs), 60)

            elif env_name == 'Cabinet':
                from envs.maniskill_env import make
                return TimeLimit(make(**kwargs), 60)
            
            elif env_name == 'FixArm':
                from envs.maniskill.fixed_arm import FixArm
                return TimeLimit(FixArm(**kwargs), 60)
 
            elif env_name == 'CabinetDense':
                from envs.maniskill_env import make
                print(f"this is dense Cabinet")
                return TimeLimit(make(obs_dim=0, reward_type="dense"), 60)

            elif env_name == 'EEArm':
                from envs.maniskill.ee_arm import EEArm
                return TimeLimit(EEArm(**kwargs), 60)

            elif env_name == 'AdroitHammer':
                from envs.modem.adroit import make_adroit_env
                return TimeLimit(make_adroit_env('hammer-v0', **kwargs), 125)

            elif env_name == 'AdroitDoor':
                from envs.modem.adroit import make_adroit_env
                return TimeLimit(make_adroit_env('door-v0', **kwargs), 100)

            elif env_name == 'KitchenSimple':
                from envs.modem.kitchen import KitchenMicrowaveKettleLightSliderV0
                return TimeLimit(KitchenMicrowaveKettleLightSliderV0(**kwargs, n_block=1), 100)

            elif env_name == 'Kitchen':
                from envs.modem.kitchen import KitchenMicrowaveKettleLightSliderV0
                return TimeLimit(KitchenMicrowaveKettleLightSliderV0(**kwargs), 250)

            elif env_name == 'Kitchen2':
                from envs.modem.kitchen import KitchenV2
                return TimeLimit(KitchenV2(**kwargs), 120)

            elif env_name == 'Kitchen3':
                from envs.modem.kitchen import KitchenV3
                return TimeLimit(KitchenV3(**kwargs), 200)

            elif env_name == 'Kitchen4':
                from envs.modem.kitchen import KitchenV4
                return TimeLimit(KitchenV4(**kwargs), 120)


            elif env_name == 'MWStickPull':
                from envs.modem.metaworld_envs import make_metaworld_env
                return TimeLimit(make_metaworld_env('stick-pull', **kwargs), 100)

            elif env_name == 'MWBasketBall':
                from envs.modem.metaworld_envs import make_metaworld_env
                return TimeLimit(make_metaworld_env('basketball', **kwargs), 100)

            elif env_name == 'PegInsert':
                #from envs.mani. import make_metaworld_env
                from envs.maniskill.peg_insert import PegInsert
                return TimeLimit(PegInsert(**kwargs), 100)

            elif env_name == 'Rope':
                from envs.softbody.plb_envs import RopeEnv
                return TimeLimit(RopeEnv(**kwargs), 50)

            elif env_name == 'AntPush':
                from envs.ant_envs import AntHEnv
                return TimeLimit(AntHEnv(env_name, **kwargs), 400)

            elif env_name == 'AntPushDense':
                from envs.ant_envs import AntHEnv
                print(f"this is dense antpush")
                return TimeLimit(AntHEnv('AntPush', obs_dim=0, reward_type="dense"), 400)

            elif env_name == 'AntPush2':
                from envs.ant_envs import AntHEnv
                return TimeLimit(AntHEnv('AntPush', **kwargs), 200)

            elif env_name == 'AntMaze':
                from envs.ant_envs import AntHEnv
                return TimeLimit(AntHEnv(env_name, **kwargs), 200)


            elif env_name == 'AntFall':
                from envs.ant_envs import AntHEnv
                return TimeLimit(AntHEnv(env_name, **kwargs), 400)


            elif env_name == 'PixelCheetah':
                from envs.mujoco_env import make
                return make('HalfCheetah-v3', **kwargs)

            return gym.make(env_name)

        self.nenv = n
        self.vec_env = SubprocVectorEnv([make_env for i in range(n)])

        self._reset = False
        self.obs = None
        self.steps = None
        self.returns = None
        self.ignore_truncated_done = ignore_truncated_done

        self.observation_space = self.vec_env.observation_space[0]
        self.action_space = self.vec_env.action_space[0]

        self.max_time_steps = self.vec_env._max_episode_steps[0]

    @property
    def anchor_state(self):
        return self.vec_env.anchor_state

    def start(self, reset=False, **kwargs):
        self.kwargs = kwargs

        if self._reset and not reset:
            pass
        else:
            self._reset = True
            self.obs = self.vec_env.reset(**kwargs) # reset all
            self.steps = np.zeros(len(self.obs), dtype=np.int64)
            self.returns = np.zeros(len(self.obs), dtype=np.float64)

        return self.obs, self.steps.copy()

    def render(self, mode='rgb_array'):
        return self.vec_env.render(id=0, mode=mode)

    def step(self, actions):
        assert self.obs is not None, "must start before running"

        actions = tonumpy(actions)
        next_obs, reward, done, info = self.vec_env.step(actions)
        end_envs = np.where(done)[0]

        import copy
        obs = copy.copy(next_obs)

        self.steps += 1
        self.returns += np.array(reward)

        episode = []
        if len(end_envs) > 0:
            if self.ignore_truncated_done:
                for j in end_envs:
                    if 'TimeLimit.truncated' in info[j]: 
                        done[j] = False

            for j in end_envs:
                step, total_reward = self.steps[j], self.returns[j]
                self.steps[j] = 0
                self.returns[j] = 0
                episode.append({'step': step, 'reward': total_reward})
                if 'success' in info[j]:
                    episode[-1]['success'] = info[j]['success']

            for idx, k in zip(end_envs, self.vec_env.reset(end_envs, **self.kwargs)):
                obs[idx] = k
        self.obs = obs.copy()

        return {
            'obs': obs, # the current observation of the environment. 
            'next_obs': next_obs, # ending state of the previous transition.
            'next_timestep': self.steps.copy(),
            'r': np.array(reward)[:, None],
            'done': done.copy(),
            'info': info,
            'total_reward': self.returns.copy(),
            'truncated': [('TimeLimit.truncated' in i) for i in info],
            'episode': episode
        }
        

    def render_traj(self, traj, **kwargs):
        # from .traj import Trajectory
        # traj: Trajectory
        # obs = traj.old_obs
        # if 'z' in traj.traj[0]:
        #     if traj.traj[0]['z'] is not None:
        #         assert 'z' not in kwargs
        #         kwargs['z'] = traj.get_tensor('z')
        # if 'a' in traj.traj[0]:
        #     assert 'a' not in kwargs
        #     kwargs['a'] = traj.get_tensor('a')
        return self.vec_env._render_traj_rgb(0, traj=traj, **kwargs)[0]

class TorchEnv(VecEnv):
    def __init__(self, env_name, n, ignore_truncated_done=False, **kwargs):
        # by default we do not have a truncated reward. 
        super().__init__()
        self.env_name = env_name
        from solver.envs import GoalEnv
        self.nenv = n

        if env_name == 'TripleMove':
            import solver.envs.softbody.triplemove
            self.goal_env: GoalEnv = GoalEnv.build(TYPE=env_name, **kwargs)

        elif env_name == 'LargeMaze':
            from envs.maze import LargeMaze
            self.goal_env = LargeMaze(**kwargs)

        elif env_name == 'SmallMaze':
            from envs.maze import SmallMaze
            self.goal_env = SmallMaze(**kwargs)

        elif env_name == 'SmallMaze2':
            from envs.maze import SmallMaze
            self.goal_env = SmallMaze(**kwargs, action_scale=0.3)

        elif env_name == 'TreeMaze':
            from envs.maze import TreeMaze
            self.goal_env = TreeMaze(**kwargs)

        elif env_name == 'GapMaze':
            from envs.maze import CrossMaze
            self.goal_env = CrossMaze(**kwargs)


        elif env_name == 'MediumMaze':
            from envs.maze import MediumMaze
            self.goal_env = MediumMaze(**kwargs)


        elif env_name == 'MediumMazeR':
            from envs.maze import MediumMaze
            self.goal_env = MediumMaze(**kwargs, reward_mapping=[[[6, 1], 10], [[6, 6], 20]])

        else:
            raise NotImplementedError

        self._reset = False
        self.obs = None
        self.steps = None
        self.returns = None
        self.ignore_truncated_done = ignore_truncated_done

        self.observation_space = self.goal_env.observation_space
        self.action_space = self.goal_env.action_space
        self.max_time_steps = self.goal_env._cfg.low_steps

    def start(self, reset=False, **kwargs):
        import torch
        self.kwargs = kwargs
        self.kwargs.update({'batch_size': self.nenv})

        if self._reset and not reset:
            pass
        else:
            self._reset = True
            self.obs = self.goal_env.reset(**kwargs) # reset all
            self.steps = torch.zeros(len(self.obs), dtype=torch.long, device='cuda:0')
            self.returns = torch.zeros(len(self.obs), dtype=torch.float32, device='cuda:0')
        return self.obs, self.steps.clone()

    def render(self, mode):
        return self.goal_env.render(mode)
        
    def render_traj(self, traj, **kwargs):
        from .traj import Trajectory
        traj: Trajectory
        obs = traj.old_obs
        if 'z' in traj.traj[0]:
            if traj.traj[0]['z'] is not None:
                assert 'z' not in kwargs
                kwargs['z'] = traj.get_tensor('z')
        if 'a' in traj.traj[0]:
            assert 'a' not in kwargs
            kwargs['a'] = traj.get_tensor('a')
        return self.goal_env._render_traj_rgb(obs, **kwargs)


    def step(self, actions):
        import torch
        assert self.obs is not None, "must start before running"
        next_obs, reward, done, _ = self.goal_env.step(actions)
        info = [{} for i in range(self.nenv)]
        done = self.steps == (self.max_time_steps - 1)
        end_envs = np.where(done.detach().cpu().numpy())[0]

        import copy
        obs = next_obs.clone()

        self.steps += 1
        self.returns += reward


        episode = []
        truncated = [False for i in info]
        if len(end_envs) > 0:
            assert len(end_envs) == self.nenv
            for j in end_envs:
                truncated[j] = True


                step, total_reward = int(self.steps[j]), float(self.returns[j])
                self.steps[j] = 0
                self.returns[j] = 0
                episode.append({'step': step, 'reward': total_reward})

                if 'success' in info[j]:
                    episode[-1]['success'] = info[j]['success']
            obs = self.goal_env.reset(**self.kwargs)

        if self.ignore_truncated_done:
            done[:] = False

        self.obs = obs.clone()
        
        return {
            'obs': obs, # the current observation of the environment. 
            'next_obs': next_obs, # ending state of the previous transition.
            'next_timestep': self.steps.clone(),
            'r': reward[:, None], #np.array(reward)[:, None],
            'done': done,
            'info': info,
            'total_reward': self.returns.clone(),
            'truncated': truncated,
            'episode': episode
        }
        

    def render_traj(self, traj, **kwargs):
        return self.goal_env._render_traj_rgb(traj=traj, **kwargs)

        
def make(env_name, n, **kwargs):
    if env_name in ['LargeMaze', 'SmallMaze', 'MediumMaze', 'TripleMove', 'TreeMaze', 'MediumMazeR', 'SmallMaze2', 'GapMaze']:
        return TorchEnv(env_name, n, **kwargs)
    else:
        return GymVecEnv(env_name, n, **kwargs)