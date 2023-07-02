import numpy as np
import torch
import solver.envs.rigidbody2d.r2d_base as r2d_base
import solver.envs.rigidbody2d.rigid2d as r2d
import gym.spaces
from tools.utils import logger
from solver.train_rpg import Trainer
from solver.envs.numerical.test_bandit_elbo import show_curve, print_elbo
from solver.envs.rigidbody2d.hooks import record_rollout, print_gpu_usage, save_traj



class Move3(r2d_base.Rigid2dBase):

    def __init__(self, cfg=None, 
        low_steps=8, dt=0.02, frame_skip=4, 
        friction_coeff=8, n_batches=32, 
        X_OBS_MUL=5.0, V_OBS_MUL=5.0, A_ACT_MUL=1.0
    ) -> None:
        super().__init__(cfg, low_steps, dt, frame_skip, friction_coeff, n_batches, X_OBS_MUL, V_OBS_MUL, A_ACT_MUL)
        self.observation_space = gym.spaces.Box(-1, 1, (4 + 2, ))

    def get_colors(self):
        return np.array([
            [255,   0,   0],
        ])

    def sample_state_goal(self, batch_size):
        state = r2d.tensor([
            [   0,    0, 0, 0,  0.3,   1], # actor

        ], batch_size=self.batch_size)
        goal_dist = 4
        goals = r2d.tensor(
            [
                0                      , -goal_dist, 
                goal_dist/2 * 3 ** 0.5 , goal_dist/2,
                -goal_dist/2 * 3 ** 0.5, goal_dist/2
            ], 
            self.batch_size
        )
        return state, goals
    
    def get_obs(self):
        obs = torch.cat([
            self.circles.state[:, :2, 0:2] / self.X_OBS_MUL,
            self.circles.state[:, :2, 2:4] / self.V_OBS_MUL
        ], dim=-1)
        return torch.cat([
            obs.view(self.batch_size, -1), 
            r2d.tensor([0, 0], batch_size=self.n_batches)
        ], dim=-1)

    def get_reward(self, s, a, s_next):
        if self.t != self.low_steps - 1:
            return torch.zeros(self.n_batches, dtype=r2d.DTYPE, device=r2d.DEVICE)
        g = self.goals.reshape(self.n_batches, -1, 2) # goals
        x = s_next[:, :1, 0:2]
        return -(x - g).norm(dim=-1).min(dim=-1)[0] ** 2

    def _render_traj_rgb(self, states, to_plot=None, **kwargs):
        """ plot actor in traj distribution plot """
        return super()._render_traj_rgb(states, to_plot, **kwargs)



def main():
    device = "cuda"
    r2d.DEVICE = device
    trainer = Trainer.parse(

        env=dict(
            TYPE='Move3',
            n_batches=32 # 128
        ),

        actor=dict(
            not_func=True,
            a_head=dict(
                TYPE='Normal', 
                linear=True, 
                squash=False, 
                std_mode='statewise', 
                std_scale=0.2)
        ),

        # RPG
        rpg=dict(
            gd=True,
            weight=dict(
                reward=0.2,
                mutual_info=10.0,
                ent_z=1.0,
                ent_a=1.0
            ),
            gamma=1.0
        ),
        # increasing_reward=1,
        z_dim=5, z_cont_dim=0,
        max_epochs=20,
        record_gif_per_epoch=1,
        path="exp/move3/gd",
        # device=device,
        format_strs="csv+stdout"
    )

    print("""========== main ==========""")
    trainer.epoch_hooks.append(print_gpu_usage)
    trainer.epoch_hooks.append(save_traj)
    # trainer.epoch_hooks.append(show_curve)
    # trainer.epoch_hooks.append(print_elbo)
    trainer.epoch_hooks.append(record_rollout)
    trainer.start()


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    main()

    # env = Move3()
    # env.reset()
    # import cv2
    # cv2.imwrite("debug.png", env.render())