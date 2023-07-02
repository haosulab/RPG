import numpy as np
from solver.envs.rigidbody2d import r2d_base
from solver.envs.rigidbody2d import rigid2d as r2d
from tools.utils import logger
from solver.train_rpg import Trainer
from solver.envs.numerical.test_bandit_elbo import show_curve, print_elbo
from solver.envs.rigidbody2d.hooks import record_rollout, print_gpu_usage, save_traj


class CenterObstaclePush(r2d_base.Rigid2dBase):

    def __init__(
        self, cfg=None, 
        low_steps=100, dt=0.01, frame_skip=0, 
        friction_coeff=3, n_batches=32, 
        X_OBS_MUL=5.0, V_OBS_MUL=5.0, A_ACT_MUL=6.0
    ) -> None:
        super().__init__(
            cfg, low_steps, dt, frame_skip, 
            friction_coeff, n_batches, 
            X_OBS_MUL, V_OBS_MUL, A_ACT_MUL
        )

    def get_colors(self):
        if not hasattr(self, "color"):
            self.color = np.array([
                [255,   0,   0],
                [  0,   0, 255],
                [  0,   0,   0],
            ])
        return self.color

    def sample_state_goal(self, batch_size):
        state = r2d.tensor([
            [-2.7,  0, 2, 0, 0.17,   1], # actor
            [-2.1,  0, 0, 0,  0.3, 2.7], # object
            [   0,  0, 0, 0,  1.5, 1e6], # obstacle
        ], batch_size=self.batch_size)
        goals = r2d.tensor([4, 0], self.batch_size)
        return state, goals

    def get_reward(self, s, a, s_next):
        """ 
            reward of transition s->a->s_next
            for base env, task is push obj to goal
        """

        # Type 1
        v = s_next[:, 0, 2:4] # actor v
        v_norm = v.norm(dim=-1)
        v_penalty = v_norm / (4 * self.V_OBS_MUL)
        # v_penalty = v_penalty * (v_norm > 1)
        # # v_penalty = 0

        # g  = self.goals        # goal
        # # x  = s     [:, 1,  :2]  # object x
        # x_ = s_next[:, 1,  :2]  # object x_
        # # t  = s     [:, 0,  :2]  # actor x
        # t_ = s_next[:, 0,  :2]  # actor x_
        # # v  = s     [:, 0, 2:4]

        # dist_to_goal = ((x_ - g)).norm(dim=-1) / (2 * self.X_OBS_MUL)
        # dist_to_blue = ((t_ - x_)).norm(dim=-1) / (2 * self.X_OBS_MUL)
        # r = - dist_to_goal - v_penalty - dist_to_blue

        # Type 2
        x_ball       = s     [:, 1, :2]
        x_ball_next  = s_next[:, 1, :2]
        x_actor      = s     [:, 0, :2]
        x_actor_next = s_next[:, 0, :2]

        actor_to_ball = (x_ball - x_actor).norm(dim=-1)
        actor_to_ball_next = (x_ball_next - x_actor_next).norm(dim=-1)
        ds_actor_to_ball = actor_to_ball - actor_to_ball_next

        closer_to_goal = (x_ball - self.goals).norm(dim=-1) - (x_ball_next - self.goals).norm(dim=-1)

        r = (- v_penalty * self.dt + (ds_actor_to_ball * (actor_to_ball > 0.6) + closer_to_goal))
        return r * 10


def main():
    r2d.DEVICE = "cuda"

    trainer = Trainer.parse(

        env=dict(
            TYPE='CenterObstaclePush',
            n_batches=100
        ),

        actor=dict(
            not_func=False,
            a_head=dict(
                TYPE='Normal', 
                linear=True, 
                squash=False, 
                std_mode='fix_no_grad', 
                std_scale=0.1)
        ),

        # RPG
        rpg=dict(
            gd=True,
            weight=dict(
                reward=1.0,
                prior=0.0
            )
        ),
        z_dim=2, z_cont_dim=0,
        max_epochs=200, n_batches=100,
        record_gif_per_epoch=1,
        path="exp/push1/rpg"
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
