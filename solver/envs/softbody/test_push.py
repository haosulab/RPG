import numpy as np
from solver.envs.rigidbody2d import r2d_base
import solver.envs.rigidbody2d.rigid2d as r2d
from tools.utils import logger
from solver.train_rpg import Trainer
from solver.envs.numerical.test_bandit_elbo import show_curve, print_elbo
from solver.envs.rigidbody2d.hooks import record_rollout, print_gpu_usage


class CenterObstaclePush(r2d_base.Rigid2dBase):

    def __init__(
        self, cfg=None, 
        low_steps=50, dt=0.01, frame_skip=2, 
        friction_coeff=10, n_batches=32, 
        X_OBS_MUL=5., V_OBS_MUL=5., A_ACT_MUL=6.0,
        no_obstacle=False,
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
            [-2.1,  0, 0, 0,  0.3,  2.7], # object
            [   0. + int(self._cfg.no_obstacle) * 100.,  0, 0, 0,  1.5, 1e6], # obstacle
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

        # Type 2
        x_ball       = s     [:, 1, :2]
        x_ball_next  = s_next[:, 1, :2]
        x_actor      = s     [:, 0, :2]
        x_actor_next = s_next[:, 0, :2]
        # print(x_ball_next[0], 'ball|actor', x_actor_next[0])

        closer_to_goal = (x_ball - self.goals).norm(dim=-1) - (x_ball_next - self.goals).norm(dim=-1)

        actor_to_ball_next = (x_ball_next - x_actor_next).norm(dim=-1)
        r = ((- v_penalty * self.dt - actor_to_ball_next) * 100. + closer_to_goal * 10) * 0.05
        return r 


class Preprocess:
    def __call__(self, s):
        s_new = s.clone()
        # print(s_new.shape)
        print(s_new[-1, 0])
        s_new[:, :, :4] = 0
        s_new[:, :, 8:] = 0
        return s_new

def main(**kwargs):
    r2d.DEVICE = "cuda"

    trainer = Trainer.parse(
        return_state=True,

        env=dict(
            TYPE='CenterObstaclePush',
            n_batches=1024,
            # no_obstacle=True,
        ),

        actor=dict(
            not_func=False,
            a_head=dict(
                TYPE='Normal', 
                linear=True, 
                squash=False, 
                std_mode='fix_no_grad', 
                std_scale=0.08)
        ),

        # RPG
        rpg=dict(
            gd=True,
            # optim =dict(lr=0.001),
        ),

        # info_net=dict(action_weight=0.),

        z_dim=10, z_cont_dim=0,
        max_epochs=1000,
        record_gif_per_epoch=1,
        n_batches=50,
        path="exp/push1/rpg",
        _update=kwargs,

        increasing_reward=0,
    )

    trainer.rpg.info_log_q.preprocess = Preprocess() # hook to preprocess the state

    print("""========== main ==========""")
    trainer.epoch_hooks.append(print_gpu_usage)
    trainer.epoch_hooks.append(show_curve)
    # trainer.epoch_hooks.append(print_elbo)
    trainer.epoch_hooks.append(lambda x, y: record_rollout(x, y, 100))
    trainer.start()


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    #main(rpg=dict(weight=dict(reward=1., ent_z=200.)))
    #main(rpg=dict(weight=dict(reward=1., ent_z=200.), gd=False), path='exp/push1/pg')
    # main(rpg=dict(weight=dict(reward=1., ent_z=200.), gd=False), z_dim=1, path='exp/push1/pure_pg')
    # main(rpg=dict(weight=dict(reward=1., ent_z=200.), gd=False), env=dict(n_batches=50), z_dim=1, path='exp/push1/pure_pg2')
    #main(rpg=dict(weight=dict(reward=1., ent_z=200.), gd=True), actor=dict(a_head=dict(std_scale=0.01)), path='exp/push1/low_std')
    #  main(rpg=dict(weight=dict(reward=1., ent_z=200.), gd=True), actor=dict(a_head=dict(std_scale=0.01)), path='exp/push1/low_std')
    main(rpg=dict(weight=dict(reward=0.2, ent_z=200., ent_a=10., mutual_info=10.), gd=True, stop_pg=True), actor=dict(a_head=dict(std_scale=0.08)), path='exp/push1/low_std', z_cont_dim=0, z_dim=5, env=dict(n_batches=50))
    # main(rpg=dict(weight=dict(reward=0.2, ent_z=200.), gd=False), env=dict(n_batches=50), z_dim=1, path='exp/push1/pure_pg3')