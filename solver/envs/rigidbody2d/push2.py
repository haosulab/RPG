import numpy as np
import solver.envs.rigidbody2d.r2d_base as r2d_base
import solver.envs.rigidbody2d.rigid2d as r2d
from tools.utils import logger
from solver.train_rpg import Trainer
from solver.envs.numerical.test_bandit_elbo import show_curve, print_elbo
from solver.envs.rigidbody2d.hooks import record_rollout, print_gpu_usage, save_traj


from solver.envs.rigidbody2d.push1 import CenterObstaclePush

class FourObstaclePush(CenterObstaclePush):

    def __init__(self, cfg=None, 
    low_steps=100, dt=0.01, frame_skip=0, 
    friction_coeff=3, n_batches=32, 
    X_OBS_MUL=5.0, V_OBS_MUL=5.0, A_ACT_MUL=6.0) -> None:
        super().__init__(cfg, low_steps, dt, frame_skip, friction_coeff, n_batches, X_OBS_MUL, V_OBS_MUL, A_ACT_MUL)

    def get_colors(self):
        if not hasattr(self, "color"):
            self.color = np.array([
                [255,   0,   0],
                [  0,   0, 255],

                [  0,   0,   0],
                [  0,   0,   0],
                [  0,   0,   0],
                [  0,   0,   0],
            ])
        return self.color

    def sample_state_goal(self, batch_size):
        state = r2d.tensor([
            [-3.5, -3.5, 1.4, 1.4, 0.17,   1], # actor
            [-3.1, -3.1,    0,  0,  0.3, 2.7], # object

            [  2, -2, 0, 0,  1.0, 1e6], # obstacle
            [ -2, -2, 0, 0,  1.0, 1e6], # obstacle
            [ -2,  2, 0, 0,  1.0, 1e6], # obstacle
            [  2,  2, 0, 0,  1.0, 1e6], # obstacle
        ], batch_size=self.batch_size)
        goals = r2d.tensor([4, 4], self.batch_size)
        return state, goals


def main():
    r2d.DEVICE = "cuda"

    trainer = Trainer.parse(

        env=dict(
            TYPE='FourObstaclePush',
            n_batches=1024
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
        ),
        z_dim=4, z_cont_dim=0,
        max_epochs=1000,
        record_gif_per_epoch=1,
        path="exp/push2.rpg"
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
