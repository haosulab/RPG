# common hooks; like saving the trajectories and so on
import numpy as np
import abc
from typing import List
from tools.config import Configurable
from tools.utils import logger


class HookBase:
    # one can config the hooks later .
    def __init__(self) -> None:
        pass

    def init(self, trainer):
        pass

    def on_epoch(self, trainer, **locals_):
        pass

HOOKS = {}
def as_hook(f):
    global HOOKS
    HOOKS[f.__name__] = f
    return f


class RLAlgo(abc.ABC):
    def __init__(self, obs_rms, hooks) -> None:
        super().__init__()
        self.epoch_id = 0
        self.total = 0

        self.obs_rms = obs_rms

        self.mode='training'


        self.hooks = hooks
        for i in hooks:
            i.init(self)

    def start(self, env, **kwargs):
        obs, timestep = env.start(**kwargs)
        obs = self.norm_obs(obs, True)
        return obs, timestep

    def step(self, env, action):
        data = env.step(action)
        # if 'success' in data['info']:
        #     data['success'] = data['info'].pop('success')
        obs = self.norm_obs(data.pop('obs'), True)
        data['next_obs'] = self.norm_obs(data['next_obs'], False)
        return data, obs

    def norm_obs(self, x, update=True):
        if self.obs_rms:
            if update and self.mode == 'training':
                self.obs_rms.update(x) # always update during training...
            x = self.obs_rms.normalize(x)

        if isinstance(x[0], dict):
            x = list(x)  # don't know what happend ..
        return x

    def sample(self, p):
        if self.mode == 'training':
            return p.sample() #TODO: not differentiable now.
        elif self.mode == 'sample':
            return p.sample()
        raise NotImplementedError


    # @abc.abstractmethod
    # def modules(self, **kwargs):
    #     pass

    @abc.abstractmethod
    def evaluate(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def inference(self, *args, **kwargs):
        pass

    def call_hooks(self, locals_):
        self.mode = 'sample'

        traj = locals_['traj']
        self.total += traj.n
        #print(self.total, )
        logger.logkvs_mean(traj.summarize_epsidoe_info())
        logger.logkv_mean('total_steps', self.total)

        self.epoch_id += 1
        locals_.pop('self')

        for i in self.hooks:
            i.on_epoch(self, **locals_)
        
        self.mode = 'training'

        
def build_hooks(hook_cfg: dict) -> List[HookBase]:
    hooks = []
    if hook_cfg is not None:
        for k, v in hook_cfg.items():
            hooks.append(HOOKS[k](**v))
    return hooks


@as_hook
class save_model(HookBase):
    def __init__(self, n_epoch=10, model_name='model'):
        self.n_epoch = n_epoch
        self.model_name = model_name

    def init(self, trainer):
        self.modules = trainer.modules()

    def on_epoch(self, trainer, **locals_):
        #return super().on_epoch(trainer, **locals_)
        if trainer.epoch_id % self.n_epoch == 0:
            logger.torch_save(self.modules, self.model_name)

    
@as_hook
class log_info(HookBase):
    def __init__(self, n_epoch=10) -> None:
        super().__init__()
        self.n_epoch = n_epoch

    def on_epoch(self, trainer, **locals_):
        if trainer.epoch_id % self.n_epoch == 0:
            logger.dumpkvs()

    

import matplotlib.pyplot as plt
class Drawer:
    def __init__(self, data):
        self.data = data
        self.images = {}
        import numpy as np
        self.background = data.get('background', {})

    def clear(self, use_bg=True):
        background = self.background
        plt.clf()
        if use_bg:
            if 'image' in background and background['image'] is not None:
                plt.imshow(np.uint8(background['image']*255))
            if 'xlim' in background:
                plt.xlim(background['xlim'])
            if 'ylim' in background:
                plt.ylim(background['ylim'])

    def get(self, name):
        from tools.utils import plt_save_fig_array
        plt.title(name)
        plt.tight_layout()
        self.images[name] = plt_save_fig_array()[:, :, :3]


@as_hook
class save_train_occupancy(HookBase):
    def __init__(self, n_epoch=10) -> None:
        super().__init__()
        self.n_epoch = n_epoch
        self.history = None 
        self.obs = []
        self.episode_metrics = {}
        
    def on_epoch(self, trainer, env, steps, traj, **locals_):
        self.obs.append(traj.get_tensor('obs').detach())

        for i in traj.traj:
            for j in range(traj.nenv):
                if i['truncated'][j] or i['done'][j]:
                    if 'metric' in i['info'][j]:
                        for k, v in i['info'][j]['metric'].items():
                            if k not in self.episode_metrics:
                                self.episode_metrics[k] = []
                            self.episode_metrics[k].append(v)

        if trainer.epoch_id % int(self.n_epoch) == 0:
            import torch
            from tools.utils import logger
            data = env.render_traj({'next_obs': torch.cat(self.obs)}, occ=True, history=self.history)
            if 'occupancy' in data['image']:
                img = data['image']['occupancy']
                logger.savefig('train_occupancy', img)

            self.history = data['history']
            
            for k, v in data['metric'].items():
                logger.logkv_mean('train_{}_metric'.format(k), v)

            for k, v in self.episode_metrics.items():
                logger.logkv_mean('train_{}_metric'.format(k), np.mean(v))

            self.obs = []
            self.episode_metrics = {}


    
@as_hook
class save_traj(HookBase):
    def __init__(self, n_epoch=10, traj_name='traj', save_gif_epochs=0, occupancy=-1, **kwargs) -> None:
        super().__init__()
        self.n_epoch = n_epoch
        self.traj_name = traj_name
        self.kwargs = kwargs
        self.save_gif_epochs = save_gif_epochs
        self.occupancy = occupancy

        self.imgs = []
        self.history = None


    def on_epoch(self, trainer: RLAlgo, env, steps, **locals_):
        """

        traj is a Trajectory
        traj.traj[0] is a dict of transition. 
        traj.traj[i]['_attrs']  records dict of values for rendering for state; must be numpy or float


        render_traj calls envs._render_traj_rgb function to generate a dict of form

        {  
            'state': np.array, T, B, 2,
            'background': {
                'image': background image (that can be plt.imshow),
                'xlim': [0, 256] or None,
                'ylim': [0, 256] or None,
            },
            'actions': np.array, T, B, 2,

            'history': (things to store)
            'images': (things to render)
            'scale': things to plot 
        }
        """
        if trainer.epoch_id % self.n_epoch == 0:
            assert trainer.mode == 'sample'
            from .traj import Trajectory
            traj: Trajectory = trainer.evaluate(env, steps)

            logger.logkvs_mean(
                {'eval_' + k: v 
                 for k, v in traj.summarize_epsidoe_info().items()}
            )

            # old_obs = traj.get_tensor('obs')
            # if trainer.obs_rms:
            #     old_obs = trainer.obs_rms.unormalize(old_obs)
            #     raise NotImplementedError
            # traj.old_obs = old_obs

            # data = env.render_traj(traj, occ=self.occupancy, history=self.history)
            # drawer = Drawer(data)

            from solver.draw_utils import plot_colored_embedding, plot_point_values
            from tools.utils import plt_save_fig_array
            # # import matplotlib.pyplot as plt

            # images = {}
            # import numpy as np
            # background = data.get('background', {})

            # def clear(use_bg=True):
            #     plt.clf()
            #     if use_bg:
            #         if 'image' in background and background['image'] is not None:
            #             plt.imshow(np.uint8(background['image']*255))
            #         if 'xlim' in background:
            #             plt.xlim(background['xlim'])
            #         if 'ylim' in background:
            #             plt.ylim(background['ylim'])

            # def get(name):
            #     plt.title(name)
            #     plt.tight_layout()
            #     images[name] = plt_save_fig_array()[:, :, :3]

            # plot z.
            # z = traj.get_tensor('z', device='cpu')
            # import torch
            # if z.dtype == torch.int64:
            #     print('zbin', torch.bincount(z.long().flatten()))

            # if 'state' in data:
            #     if not isinstance(data['state'], dict):
            #         print('z.shape', z.shape, 'data state shape', data['state'].shape)
            #         drawer.clear()
            #         plot_colored_embedding(z, data['state'], s=2)
            #         drawer.get('latent')
            #     else:
            #         for k, v in data['state'].items():
            #             drawer.clear()
            #             plot_colored_embedding(z, v, s=2)
            #             drawer.get(k)

            # if 'actions' in data:
            #     drawer.clear(use_bg=False)
            #     plot_colored_embedding(z, data['actions'])
            #     drawer.get('action')

            # if 'image' in data:
            #     for k, v in data['image'].items():
            #         drawer.clear(use_bg=False)
            #         plt.imshow(v)
            #         plt.axis('off')
            #         drawer.get(k)

            # if 'metric' in data:
            #     for k, v in data['metric'].items():
            #         logger.logkv_mean(f'test_{k}_metric', v)

            # if 'history' in data:
            #     self.history = data['history']


                
            # # traj _attrs, record 
            # if '_attrs' in traj.traj[0] and ('state' in data and not isinstance(data['state'], dict)):
            #     others = []
            #     for k in traj.traj[0]['_attrs']:
            #         v = [i['_attrs'][k] for i in traj.traj]
            #         v = np.array(v)
            #         print(k, v.shape)

            #         drawer.clear()
            #         v = v.reshape(-1)
            #         plot_point_values(v.reshape(-1), data['state'], s=2)
            #         drawer.get(k)
            #         others.append(drawer.images.pop(k))
            #     drawer.images['attrs'] = np.concatenate(others, axis=1)

            # for k, v in drawer.images.items():
            #     logger.savefig(k + '.png', v)

            # #logger.savefig(self.traj_name + '.png', img)
            # if self.save_gif_epochs > 0 and len(drawer.images) > 0:
            #     img = np.concatenate([v for k, v in drawer.images.items()], axis=1)
            #     self.imgs.append(img)
            #     if len(self.imgs) % self.save_gif_epochs == 0:
            #         logger.animate(self.imgs, self.traj_name + '.mp4')


@as_hook
class evaluate_pi(HookBase):
    def __init__(self, n_epoch=10, **kwargs) -> None:
        super().__init__()
        self.n_epoch = n_epoch
        self.kwargs = kwargs

    def on_epoch(self, trainer: RLAlgo, env, steps, **locals_):
        if trainer.epoch_id % self.n_epoch == 0:
            assert trainer.mode == 'sample'
            traj = trainer.evaluate(env, steps)
            from tools.utils import logger
            logger.logkvs_mean(
                {'eval_' + k: v 
                 for k, v in traj.summarize_epsidoe_info().items()}
            )
                    


@as_hook
class monitor_action_std(HookBase):
    def __init__(self, n_epoch=1, std_decay=None) -> None:
        super().__init__()
        self.n_epoch = n_epoch
        self.std_decay = std_decay

    def init(self, trainer):
        self.head = trainer.pi.actor_optim.actor.head
        assert self.head.std_mode.startswith('fix')

        if self.std_decay is not None:
            from tools.utils import scheduler
            self.scheduler = scheduler.Scheduler.build(
                cfg=self.std_decay, init_value=self.head.std_scale)

    def on_epoch(self, trainer, **locals_):
        if trainer.epoch_id % self.n_epoch == 0:
            import torch
            print('action_std', (torch.exp(self.head.log_std) * self.head.std_scale).detach().cpu().numpy().reshape(-1))

            if self.std_decay is not None:
                self.head.std_scale = self.scheduler.step(epoch=trainer.total)



@as_hook
class plot_anchor(HookBase):
    def __init__(self, n_epoch=1) -> None:
        super().__init__()
        self.n_epoch = n_epoch

    def init(self, trainer):
        self.env = trainer.env
        info = self.env.anchor_state[0]
        self.state = info['state']
        self.coord = info['coord']

    def on_epoch(self, trainer, **locals_):
        from solver.draw_utils import plot_point_values
        if trainer.epoch_id % self.n_epoch == 0:
            rnd = trainer.exploration.intrinsic_reward(self.state).detach().cpu().numpy()
            import matplotlib.pyplot as plt
            plt.clf()
            plot_point_values(rnd, self.coord, s=2)
            logger.savefig('rnd.png')

            
            
            
@as_hook
class accumulate_ant_traj(HookBase):
    # just for test_offline_ant.sh
    def __init__(self, n_epoch=10) -> None:
        super().__init__()
        #self.open = open
        self.n_epoch = int(n_epoch)
        self.n_succ = np.zeros((4, 4), dtype=np.int64)
        self.n_total = np.zeros((4, 4), dtype=np.int64)

    def on_epoch(self, trainer, traj, **locals_):
        from .traj import Trajectory
        traj: Trajectory = traj
        obs = traj.get_tensor('obs')
        assert obs.shape[1] == 1
        _, truncated = traj.get_truncated_done()
        grid = obs[:, 0, :2] * 100. * 4
        s0 = grid[0]
        x, y = int(s0[0]), int(s0[1])
        import torch
        reward = torch.logical_and((torch.abs(grid[..., 0] - 0.5) < 0.5), (torch.abs(grid[..., 1] - 3.5) < 0.5))[..., None]
        success = float(reward.any())
        self.n_total[x, y] += 1
        self.n_succ[x, y] += int(success)
        assert truncated[-1].all()
        print(x, y, int(success), obs[0, 0, :2])

        if trainer.epoch_id % self.n_epoch == 0:
            success_rate = self.n_succ / self.n_total.clip(1, np.inf)
            print(success_rate)
            logger.savefig('success_rate.png', success_rate)
