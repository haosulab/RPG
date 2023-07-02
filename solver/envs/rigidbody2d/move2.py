import numpy as np
from tools.utils import logger
from solver.train_rpg import Trainer

import solver.envs.rigidbody2d.r2d_base as r2d_base
import solver.envs.rigidbody2d.rigid2d as r2d
from solver.envs.numerical.test_bandit_elbo import show_curve, print_elbo
from solver.envs.rigidbody2d.hooks import record_rollout, print_gpu_usage, save_traj



class FourObstaclesMove(r2d_base.Rigid2dBase):

    def __init__(self, cfg=None, 
        low_steps=20, dt=0.02, frame_skip=1, 
        friction_coeff=8, n_batches=32, 
        X_OBS_MUL=5.0, V_OBS_MUL=5.0, A_ACT_MUL=1.0
    ) -> None:
        super().__init__(cfg, low_steps, dt, frame_skip, friction_coeff, n_batches, X_OBS_MUL, V_OBS_MUL, A_ACT_MUL)

    def get_colors(self):
        return np.array([

            [255,   0,   0],
            [  0,   0,   0],
            [  0,   0,   0],
            [  0,   0,   0],
            [  0,   0,   0],

        ])

    def sample_state_goal(self, batch_size):
        state = r2d.tensor([
            [-3.7, -3.7, 2, 2, 0.17,   1], # actor
            [  2, -2, 0, 0,  1.0, 1e8], # obstacle
            [ -2, -2, 0, 0,  1.0, 1e8], # obstacle
            [ -2,  2, 0, 0,  1.0, 1e8], # obstacle
            [  2,  2, 0, 0,  1.0, 1e8], # obstacle
            
        ], batch_size=self.batch_size)
        goals = r2d.tensor([4, 4], self.batch_size)
        return state, goals

    def get_reward(self, s, a, s_next):
        """ 
            reward of transition s->a->s_next
            for base env, task is push obj to goal
        """
        # still ok
        # g = self.goals      # goal
        # t = s_next[:, 0, :2]  # actor
        # actor_to_goal  = (t - g).norm(dim=-1)
        # d = (actor_to_goal) / (2 * self.X_OBS_MUL)
        # return -d

        # try better
        g = self.goals      # goal
        t = s_next[:, 0, :2]  # actor
        actor_to_goal  = (t - g).norm(dim=-1)
        # d = (actor_to_goal) ** 2 / (4 * self.X_OBS_MUL ** 2)
        d = (actor_to_goal) / (4 * self.X_OBS_MUL)
        return -d

    def _render_traj_rgb(self, states, to_plot=None, **kwargs):
        return super()._render_traj_rgb(states, to_plot, **kwargs)



def main():
    r2d.DEVICE = "cuda"

    trainer = Trainer.parse(

        env=dict(
            TYPE='FourObstaclesMove', 
            low_steps=50, dt=0.02, 
            frame_skip=0, friction_coeff=8,
            A_ACT_MUL=2.0,
            n_batches=1024
        ),

        actor=dict(
            not_func=False,
            a_head=dict(
                TYPE='Normal', 
                linear=True, 
                squash=False, 
                std_mode='fix_no_grad', 
                std_scale=0.3)
        ),

        # RPG
        rpg=dict(
            gd=True,
            weight=dict(

            )
        ),

        z_dim=6, z_cont_dim=0,
        max_epochs=100,
        record_gif_per_epoch=5,
        path="exp/move2/rpg"
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
