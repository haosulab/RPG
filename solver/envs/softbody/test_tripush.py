import torch
import gym
import numpy as np
from solver.envs.rigidbody2d import r2d_base

from tools.utils import totensor
import solver.envs.rigidbody2d.rigid2d as r2d
from tools.utils import logger
from solver.train_rpg import Trainer
from solver.envs.numerical.test_bandit_elbo import show_curve, print_elbo
from solver.envs.rigidbody2d.hooks import record_rollout, print_gpu_usage


class Tripush(r2d_base.Rigid2dBase):

    def __init__(
        self, cfg=None, 
        low_steps=50, dt=0.01, frame_skip=2, 
        friction_coeff=3, n_batches=32, 
        X_OBS_MUL=5, V_OBS_MUL=5, A_ACT_MUL=6.0,
        no_obstacle=False,
    ) -> None:
        super().__init__(
            cfg, low_steps, dt, frame_skip, 
            friction_coeff, n_batches, 
            X_OBS_MUL, V_OBS_MUL, A_ACT_MUL
        )
        self.observation_space = gym.spaces.Box(-1, 1, (4 * 2 + 4*2 + 2,))

    def get_colors(self):
        if not hasattr(self, "color"):
            self.color = np.array([
                [255,   0,   0],
                [  0,   0, 255],
                [  0,   255, 255],
                [  0,   255, 0],
                [  0,   0,   0],
            ])
        return self.color

    def sample_state_goal(self, batch_size):
        state = r2d.tensor([
            [-4,  0, 2, 0, 0.17,   1], # actor
            [-2.1,  0, 0, 0,  0.3,  1.0], # object
            [-2.1,  2, 0, 0,  0.3,  1.0], # object
            [-2.1,  -2, 0, 0,  0.3,  1.0], # object
            [   0.5 * 4,  0.4 * 4, 0, 0,  3, 1e6], # obstacle
        ], batch_size=self.batch_size)
        goals = r2d.tensor([4, 0], self.batch_size)
        return state, goals

    def reset(self, *args, **kwargs):
        obs = super().reset(*args, **kwargs)
        self.nsteps = 0
        return obs

    
    def get_obs(self):
        return torch.cat([
            (self.circles.state[:, :4, 0:2] / self.X_OBS_MUL).reshape(self.batch_size, -1),
            (self.circles.state[:, :4, 2:4] / self.V_OBS_MUL).reshape(self.batch_size, -1),
            # obs.view(self.batch_size, -1),
            self.goals  # goals
        ], dim=-1)


    def get_reward(self, s, a, s_next):
        """ 
            reward of transition s->a->s_next
            for base env, task is push obj to goal
        """
        # Type 1
        v = s_next[:, 0, 2:4] # actor v
        v_norm = v.norm(dim=-1)
        v_penalty = v_norm / (4 * self.V_OBS_MUL)

        # Type 2
        dists = []
        togoals = []
        for i in range(3):
            # goals = []
            x_actor      = s     [:, 0, :2]
            x_actor_next = s_next[:, 0, :2]
            x_ball       = s     [:, i+1, :2]
            x_ball_next  = s_next[:, i+1, :2]

            goal = totensor([4, [0, 2, -2][i]], device='cuda:0')

            actor_to_ball_next = (x_ball_next - x_actor_next).norm(dim=-1)

            dists.append(actor_to_ball_next)

            closer_to_goal = (x_ball - goal).norm(dim=-1) - (x_ball_next - goal).norm(dim=-1)
            togoals.append(closer_to_goal)


        if self.nsteps > 20:
            dists = torch.stack(dists).min(axis=0)[0]
        else:
            dists = 0

        togoals = torch.stack(togoals).sum(axis=0)
        self.nsteps += 1

        r = -v_penalty * self.dt * 100 - dists + togoals * 10
        return r 

    def step(self, action):
        from tools.utils import clamp
        # action = clamp(action, -0.3, 0.3)
        return super().step(action)


class Preprocess:
    def __call__(self, s):
        s_new = s.clone()
        # s_new[:, 0, :2] = 0
        s_new[:, 0, 8:] = 0
        return s_new


def main(**kwargs):
    r2d.DEVICE = "cuda"

    trainer = Trainer.parse(

        env=dict(
            TYPE='Tripush',
            n_batches=500,
            # no_obstacle=True,
        ),

        actor=dict(
            not_func=False,
            a_head=dict(
                TYPE='Normal', 
                linear=True, 
                squash=False, # let's limit the velocity..
                std_mode='fix_no_grad', 
                std_scale=0.1), # 0.1 should be fine ..
            ignore_previous_z=True,
        ),

        # RPG
        rpg=dict(
            gd=True,
            weight=dict(ent_z=1.)
        ),

        info_net=dict(action_weight=0.),

        z_dim=20, z_cont_dim=0,
        max_epochs=100,
        record_gif_per_epoch=1,
        increasing_reward=0,
        record_gif=False,
        n_batches=50,
        _update=kwargs,
        path="exp/tripush/rpg",
    )

    trainer.rpg.info_log_q.preprocess = Preprocess() # hook to preprocess the state

    print("""========== main ==========""")
    trainer.epoch_hooks.append(print_gpu_usage)
    trainer.epoch_hooks.append(show_curve)
    # trainer.epoch_hooks.append(print_elbo)
    trainer.epoch_hooks.append(lambda x, y: record_rollout(x, y, batch_size=100, save_last=True))
    trainer.start()


def render():
    r2d.DEVICE = "cuda"

    from tools.utils import totensor
    env = Tripush(n_batches=1)
    #env.render()
    images = []
    env.reset()
    for i in range(10):
        env.step(totensor(env.action_space.sample()[None, :], device='cuda:0'))
        images.append(env.render('rgb_array'))
    #logger.save_video(images, "exp/push1/rpg/rollout.mp4")
    from tools.utils import animate
    animate(images, "xx.mp4")


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    #main()
    # render()
    main(rpg=dict(use_action_entropy=False))
