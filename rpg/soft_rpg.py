import tqdm
import numpy as np
from einops import repeat
import torch
from .common_hooks import RLAlgo, build_hooks
from .env_base import GymVecEnv, TorchEnv
from .buffer import ReplayBuffer
from .info_net import InfoLearner
from .traj import Trajectory
from .intrinsic import IntrinsicMotivation
#from .rnd import RNDExplorer
from .exploration_bonus import ExplorationBonus
from .utils import create_hidden_space
from nn.distributions import Normal, DistHead
from typing import Union
from tools.config import Configurable
from tools.utils import logger, totensor
from .policy_learner import PolicyLearner
from .hidden import HiddenSpace, Categorical
from .worldmodel import HiddenDynamicNet, DynamicsLearner
from . import repr 
import re

def scheduler(step, schedule):
    if schedule is None:
        return 1.
    if isinstance(schedule, str) and '2seg' in schedule:
        match = re.match(r'2seg\((.+),(.+),(.+)\)', schedule)
        score1, t1, t2 = [float(g) for g in match.groups()]

        return np.clip(step / t1, 0.0, 1.0) * score1 + np.clip((step - t1) / (t2 - t1), 0.0, 1.0) * (1-score1)

    elif isinstance(schedule, int) or 'exp' not in schedule:
        if isinstance(schedule, str):
            match = re.match(r'linear\((.+),(.+),(.+)\)', schedule)
            if match:
                init, final, duration = [float(g) for g in match.groups()]
            else:
                duration = int(schedule)
                init, final = 0, 1
        else:
            duration = int(schedule)
            init, final = 0., 1.
        mix = np.clip(step / duration, 0.0, 1.0)
        return (1.0 - mix) * init + mix * final
    else:
        match = re.match(r'exp\((.+),(.+),(.+)\)', schedule)
        if match:
            # init should be zero
            init, gamma, duration = [float(g) for g in match.groups()]
            # init value
            # gradually decay to 0
            target_step = np.maximum(duration - step, 0.)
            return init + gamma ** (target_step)
        else:
            raise NotImplementedError


class Trainer(Configurable, RLAlgo):
    def __init__(
        self,
        env: Union[GymVecEnv, TorchEnv],
        cfg=None,

        max_epoch=None,

        env_name=None,
        env_cfg=None,

        #z_dim=1, z_cont_dim=0,
        hidden=HiddenSpace.to_build(TYPE=Categorical),

        # model
        buffer=ReplayBuffer.dc,
        model=HiddenDynamicNet.dc,
        trainer=DynamicsLearner.dc,

        # policy
        pi_a=PolicyLearner.dc,
        pi_z=PolicyLearner.dc,

        head = DistHead.gdc(
            linear=False,
            squash=True,
            std_mode='statewise',
            std_scale=1.,
            TYPE='Normal',
        ),
        z_head=None,

        # mutual information term ..
        info = InfoLearner.dc,
        rnd = ExplorationBonus.dc, # rnd can be any density estimator 

        # update parameters..
        horizon=6, batch_size=512, 
        update_target_freq=2, update_train_step=1, warmup_steps=1000, steps_per_epoch=None,

        actor_delay=2, z_delay=0, info_delay=0,
        eval_episode=10, save_video=0,

        # trainer utils ..
        hooks=None, path=None, wandb=None, log_date=False,
        tau=0.005,
        relabel_latent=None,
        relabel_method='forward',

        time_embedding=0,

        seed=None,

        
        backbone=None,
        reward_scale=1.,
        reward_relabel = None,
        reward_schedule = None,
        use_reward_schedule=False,


        actor_buffer=None, # use the limit the data for training the policy network .. tend to use the latest data if necessary

        fix_buffer=None, # offline training .. 
        save_buffer_epoch=0,
        save_model_epoch=0,

        max_total_steps=None,
        save_eval_results=False,
        cem=None,
    ):
        if seed is not None:
            from tools.utils import set_seed
            set_seed(seed)

        if env is None:
            from tools.config import CN
            from .env_base import make
            env = make(env_name, **CN(env_cfg))
        self.env = env
        print('path', path)
    
        Configurable.__init__(self)
        RLAlgo.__init__(self, None, build_hooks(hooks))

        obs_space = env.observation_space
        #self.z_space = z_space = create_hidden_space(z_dim, z_cont_dim)
        #self.use_z = z_dim > 1 or z_cont_dim > 0
        z_space: HiddenSpace = HiddenSpace.build(hidden)
        self.z_space = z_space

        self.horizon = horizon
        self.device = 'cuda:0'

        # reward_with_latent = self.z_space.learn and info.use_latent

        if fix_buffer is not None:
            self.buffer = torch.load(fix_buffer)
        else:
            self.buffer = ReplayBuffer(obs_space, env.action_space, env.max_time_steps, horizon, cfg=buffer)

        reward_requires_latent=False
        if self._cfg.rnd.scale > 0. and self._cfg.rnd.include_latent:
            reward_requires_latent=True

        self.dynamics_net = HiddenDynamicNet(obs_space, env.action_space, z_space, time_embedding, cfg=model, value_backbone=backbone, reward_requires_latent=reward_requires_latent).cuda()

        state_dim = self.dynamics_net.state_dim
        hidden_dim = 256

        from .policy_net import DiffPolicy, QPolicy
        pi_a_net = DiffPolicy(state_dim, z_space.dim, hidden_dim, DistHead.build(env.action_space, cfg=head),
                              cfg=backbone, time_embedding=time_embedding).cuda()
        self.pi_a = PolicyLearner('a', env.action_space, pi_a_net, z_space.tokenize, cfg=pi_a)

        # z learning ..
        z_head = self.z_space.make_policy_head(z_head)
        
        pi_z_net = QPolicy(state_dim, hidden_dim, z_head, cfg=backbone, time_embedding=time_embedding).cuda()
        self.pi_z = PolicyLearner('z', env.action_space, pi_z_net, z_space.tokenize, cfg=pi_z, ignore_hidden=True)

        self.z = None
        self.update_step = 0

        self.intrinsics = [self.pi_a, self.pi_z]

        auxilary = None
        if self.z_space.learn:
            # learn_posterior = (relabel > 0. and relabel_method != 'future')
            self.info_learner = InfoLearner(obs_space, state_dim, env.action_space, z_space, cfg=info,
                                            hidden_dim=hidden_dim) #, learn_posterior=learn_posterior)
            print("Hidden", z_space, self.z_space.learn, self.info_learner.use_latent)
            assert self.info_learner.use_latent
            if self.info_learner.use_latent:
                self.intrinsics.append(self.info_learner)

            if self.info_learner.learn_with_dynamics:
                auxilary = self.info_learner.net


        #self.info_net = lambda x: x['rewards'], 0, 0
        # print(f"{self.dynamics_net=}")
        # print(f"{auxilary=}")
        # print(f"{self.pi_a=}")
        # print(f"{self.pi_z=}")
        self.make_rnd()
        self.model_learner = DynamicsLearner(self.dynamics_net, auxilary, self.pi_a, self.pi_z, cfg=trainer)

        self.intrinsic_reward = IntrinsicMotivation(*self.intrinsics)
        self.model_learner.set_intrinsic(self.intrinsic_reward)

        z_space.callback(self)

        if reward_relabel is not None:
            from .reward_relabel import relabel
            self.reward_relabel = lambda *args: relabel(reward_relabel, *args)
        else:
            self.reward_relabel = None

            

        # CEM part
        if cem is not None:
            from .cem import CEM
            self.cem = CEM(self.dynamics_net, horizon=horizon, action_dim=env.action_space.shape[-1], cfg=cem)
        else:
            self.cem = None
        
        # print(f"{self.exploration=}")
        # if self.exploration is not None:
        #     print(f"{self.exploration.update_freq=}")
        #     print(f"{self.exploration.update_step=}")
        # print(f"{self.z_space.obs_dim=}")
        # print(f"{self.z_space.head.std_scale=}")
        # # print(f"{pi_a_net.head=}")
        # # print(f"{pi_a_net.head.std_mode=}")
        # # print(f"{pi_a_net.head.std_scale=}")
        # print(f"{pi_z_net.head._cfg=}")
        # print(f"{pi_a_net.head._cfg=}")
        # print(f"{self.model_learner._cfg=}")
        # print(f"{self.pi_a_net._cfg=}")
        # print(f"{self.pi_a_net._cfg=}")
        # print(f"{self.pi_a_net._cfg=}")
        # print(f"{self.pi_a_net._cfg=}")
        # breakpoint()
        # exit()

    def make_rnd(self):
        rnd = self._cfg.rnd
        if rnd.scale > 0.:
            self.exploration = ExplorationBonus(
                self.env.observation_space, self.dynamics_net.state_dim, self.z_space, 
                self.dynamics_net.enc_s, self.buffer, cfg=rnd)

            if not self.exploration.as_reward:
                self.intrinsics.append(self.exploration)
        else:
            self.exploration = None

        # from IPython import embed; embed()
        # exit(0)

    def relabel_z(self, seg, relabel_z=True):
        z = seg.z
        if self._cfg.fix_buffer:
            z = torch.zeros_like(z)
            assert self._cfg.hidden.n == 1
        # .obs_seq[0], seg.timesteps[0], seg.z
        if self._cfg.relabel_latent is not None and self.update_step > 200 and relabel_z:
            relabel_ratio = scheduler(self.update_step, self._cfg.relabel_latent)

            if self._cfg.relabel_method == 'future':
                assert seg.future is not None
                o, no, a = seg.future
                traj = self.dynamics_net.enc_s(torch.stack((o, no)))
                new_z = self.info_learner.sample_z(traj, a)[0]
                raise NotImplementedError("future relabel is not implemented")
            else:
                t = seg.timesteps[0]
                # we need to compute new_z with pi_z for t == 0 or sample_z for t > 0
                is_start = (t == 0)
                states = self.dynamics_net.enc_s(seg.obs_seq[0])
                new_z = self.info_learner.sample_latent(states)
                if is_start.any():
                    new_z[is_start] = self.pi_z(states[is_start], seg.z, prev_action=None).a

            mask = torch.rand(size=(len(z),)) < relabel_ratio
            # if new_z.dtype == torch.float32:
            #     # discard samples with too low probability
            #     prior = torch.distributions.Normal(0, 1)
            #     mask[prior.log_prob(new_z).mean(axis=-1) < -1.] = False

            z = z.clone()
            z[mask] = new_z[mask]

            if new_z.dtype == torch.float32:
                logger.logkv_mean('relabel_z_range', new_z.std())
            logger.logkv_mean('relabel_ratio', mask.float().mean())

        return z

    def update_dynamcis(self):
        seg = self.buffer.sample(self._cfg.batch_size)
        # never relabel z for dynamics learning ..
        # this is the off policy part
        z = self.relabel_z(seg, relabel_z=False) # don't do relabel for dynamics learning
        
        if self.reward_relabel is not None:
            seg.reward = self.reward_relabel(seg.obs_seq[:-1], seg.action, seg.obs_seq[1:])
        
        if self._cfg.reward_schedule is not None and self._cfg.use_reward_schedule:
            reward_schedule = scheduler(
                self.update_step, self._cfg.reward_schedule
            )
            logger.logkv_mean('reward_schedule', reward_schedule)
        else:
            reward_schedule = 1.

        seg.reward = seg.reward * self._cfg.reward_scale * reward_schedule

        if self.exploration is not None and self.exploration.as_reward:
            intrinsic_reward = self.exploration.intrinsic_reward(seg.obs_seq[1:], repeat(z, "... -> t ...", t=len(seg.obs_seq) - 1))
            logger.logkv_mean('reward_rnd', intrinsic_reward.mean())
            logger.logkv_mean('reward_rnd_std', intrinsic_reward.std())
            seg.reward = seg.reward + intrinsic_reward

        # if self.z_space.learn:
        #     seg.z = z
        #     rollout = self.info_learner.seg2rollout(seg)
        #     r = self.info_learner.intrinsic_reward(rollout)[1]
        #     seg.reward = seg.reward + r
        #     logger.logkv_mean('reward_info', r.mean())
        # #   self.info_learner.update_batch(seg)

        prev_z = z[None, :].expand(len(seg.obs_seq), *seg.z.shape)
        self.model_learner.learn_dynamics(
            seg.obs_seq, seg.timesteps, seg.action, seg.reward, seg.done, seg.truncated_mask, prev_z)

        # if self.z_space.learn:
        #     self.info_learner.update_batch(seg)
    
    def update_pi_a(self):
        actor_delay = self._cfg.actor_delay
        info_delay = self._cfg.info_delay

        update_pi_a = self.update_step % actor_delay == 0
        update_info = (info_delay > 0 and self.update_step % info_delay == 0) or (info_delay == 0 and update_pi_a)

        if update_pi_a or update_info:
            seg = self.buffer.sample(self._cfg.batch_size, horizon=1, latest=self._cfg.actor_buffer)
            z = self.relabel_z(seg)
            rollout = self.dynamics_net.inference(
                seg.obs_seq[0], z, seg.timesteps[0], self.horizon, self.pi_z, self.pi_a,
                intrinsic_reward=self.intrinsic_reward
            )

            if update_pi_a:
                self.pi_a.update(rollout)
                if self.exploration is not None:
                    self.exploration.update_by_rollout(rollout)
                for k, v in rollout['extra_rewards'].items():
                    logger.logkv_mean_std(f'reward_{k}', v)

            if update_info:
                if self.z_space.learn:
                    self.info_learner.update(rollout)

    def update_pi_z(self):
        from .hidden import Gaussian
        if self._cfg.z_delay > 0 and self.update_step % self._cfg.z_delay == 0 and not isinstance(self.z_space, Gaussian):
            o, z, t = self.buffer.sample_start(self._cfg.batch_size)
            assert (t < 1).all()
            #z = self.relabel_z(o, t, z) relabel does not makes any sense here
            rollout = self.dynamics_net.inference(o, z, t, self.horizon, self.pi_z, self.pi_a, intrinsic_reward=self.intrinsic_reward)
            self.pi_z.update(rollout)

    def update(self):
        self.update_dynamcis()
        self.update_pi_a()
        self.update_pi_z()
        self.update_step += 1
        if self.update_step % self._cfg.update_target_freq == 0:
            self.model_learner.ema(self._cfg.tau)
            self.pi_a.ema(self._cfg.tau)
            self.pi_z.ema(self._cfg.tau)
            if self.z_space.learn:
                self.info_learner.ema(self._cfg.tau)

    def eval(self):
        #print("Eval is not implemented")
        # eval not implemented
        return

    def train(self):
        # train
        return

    def policy(self, obs, prevz, timestep):
        obs = totensor(obs, self.device)
        prevz = totensor(prevz, self.device, dtype=None)
        timestep = totensor(timestep, self.device, dtype=None)
        s = self.dynamics_net.enc_s(obs)
        z = self.pi_z(s, prevz, prev_action=prevz, timestep=timestep).a
        if self.cem is None:
            a = self.pi_a(s, z, timestep=timestep).a
        else:
            a = self.cem.plan(obs, timestep, z,  step=self.total, 
                              pi_a=self.pi_a, pi_z=self.pi_z, intrinsic_reward=self.intrinsic_reward)
        return a, z.detach().cpu().numpy()

    def inference(self, n_step, mode='training'):
        with torch.no_grad():
            z_space = self.z_space.space
            obs, timestep = self.start(self.env)
            if self.z is None:
                self.z = [z_space.sample()] * len(obs)
            for idx in range(len(obs)):
                if timestep[idx] == 0:
                    self.z[idx] = z_space.sample() * 0
        r = tqdm.trange if self._cfg.update_train_step > 0 else range 

        transitions = []
        images = []
        for idx in r(n_step):
            with torch.no_grad():
                self.eval()
                transition = dict(obs = obs, timestep=timestep)
                a, self.z = self.policy(obs, self.z, timestep)
                prevz = totensor(self.z, device=self.device, dtype=None)
                data, obs = self.step(self.env, a)

                if mode != 'training' and self._cfg.save_video > 0 and idx < self._cfg.save_video: # save video steps
                    images.append(self.env.render('rgb_array')[0])

                transition.update(**data, a=a, z=prevz)
                if self.reward_relabel is not None:
                    newr = self.reward_relabel(transition['obs'], transition['a'], transition['next_obs'])
                    assert newr.shape == transition['r'].shape
                    transition['r'] = newr

                if mode != 'training' and self.exploration is not None:
                    # for visualize the transitions ..
                    transition['next_state'] = self.dynamics_net.enc_s(totensor(obs, device='cuda:0'))

                    if self.exploration is not None:
                        self.exploration.visualize_transition(transition)

                transitions.append(transition)
                timestep = transition['next_timestep']
                for j in range(len(obs)):
                    if timestep[j] == 0:
                        self.z[j] = z_space.sample() * 0

            if mode == 'training' and self.exploration is not None:
                self.exploration.add_data(obs, prevz)

            if self.buffer.total_size() > self._cfg.warmup_steps and self._cfg.update_train_step > 0 and mode == 'training':
                if idx % self._cfg.update_train_step == 0:
                    self.update()

        if len(images) > 0:
            logger.animate(images, 'eval.mp4')


        traj =Trajectory(transitions, len(obs), n_step)
        if mode != 'training':
            # if hasattr(self._cfg, 'save_eval_results') and self._cfg.save_eval_results:
            #os.makedirs(self._cfg.save_eval_results, exist_ok=True)
            import os
            path = logger.dir_path(f"eval{self.epoch_id}")
            os.makedirs(path, exist_ok=True)
            # if len(images) == 0:
            #     from IPython import embed; embed()
            assert len(images) > 0
            logger.animate(images, f'eval{self.epoch_id}/video.mp4')
            torch.save(traj, os.path.join(path, 'traj.pt'))
        return traj

    def setup_logger(self):
        format_strs = ["stdout", "log", "csv"]
        kwargs = {}
        if self._cfg.wandb is not None:
            wandb_cfg = dict(self._cfg.wandb)
            if 'stop' not in wandb_cfg:
                format_strs = format_strs[:3] + ['wandb']
                kwargs['config'] = self._cfg
                kwargs['project'] = wandb_cfg.get('project', 'mujoco_rl')
                kwargs['group'] = wandb_cfg.get('group', self.env.env_name)

                try:
                    name = self._cfg._exp_name
                except:
                    name = None
                kwargs['name'] = wandb_cfg.get('name', None) + (('_' + name) if name is not None else '')

        logger.configure(dir=self._cfg.path, format_strs=format_strs, date=self._cfg.log_date, **kwargs)

        import os
        with open(os.path.join(logger.get_dir(), 'config.yaml'), 'w') as f:
            f.write(str(self._cfg))

    def run_rpgm(self):
        self.setup_logger()
        env = self.env
        max_epoch = self._cfg.max_epoch

        steps = self._cfg.steps_per_epoch or self.env.max_time_steps
        epoch_id = 0
        while True:
            if max_epoch is not None and epoch_id >= max_epoch:
                break
            self.epoch_id = epoch_id

            traj = self.inference(steps)

            if self._cfg.fix_buffer is None:
                self.buffer.add(traj)

            a = traj.get_tensor('a')
            logger.logkv('epoch', epoch_id)
            logger.logkv_mean('a_max', float(a.max()))
            logger.logkv_mean('a_min', float(a.min()))

            self.call_hooks(locals())
            print(traj.summarize_epsidoe_info())

            logger.dumpkvs()
            epoch_id += 1

            if self._cfg.save_buffer_epoch > 0 and epoch_id % self._cfg.save_buffer_epoch == 0:
                logger.torch_save(self.buffer, 'buffer.pt')

            if self._cfg.save_model_epoch > 0 and epoch_id % self._cfg.save_model_epoch == 0 or epoch_id == 0:
                env = self.env 
                self.env = None
                logger.torch_save(self, f'model{epoch_id}.pt')
                self.env = env

            if self._cfg.max_total_steps is not None: 
                if self.total >= self._cfg.max_total_steps:
                    break

    def evaluate(self, env, steps):
        with torch.no_grad():
            self.start(self.env, reset=True)
            out = self.inference(steps * self._cfg.eval_episode, mode='evaluate')
            self.start(self.env, reset=True)
            return out