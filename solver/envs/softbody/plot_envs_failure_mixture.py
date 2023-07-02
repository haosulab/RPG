import torch
import numpy as np
from tools.utils import logger, totensor
import solver.envs.softbody
from solver.envs.softbody import TripleMove
from solver.train_rpg import Trainer
from solver.envs.numerical.test_bandit_elbo import show_curve, print_elbo 

class TripleCont(TripleMove):
    def __init__(self, cfg=None, boundary=False):
        super().__init__()

        self.goals = totensor(
            [
                [0, 0.37 ** 0.5], 
                [1./2 * 3 ** 0.5 , 1./2],
                [-1./2 * 3 ** 0.5, 1./2]
            ], 
            device='cuda:0'
        )[:self._cfg.n_goals] * 0.4 + 0.5

        self.goals[0, 1] = 0.37 ** 0.5 + 0.2

        print(self.goals)

    def sample_state_goal(self, batch_size=1):
        state = np.zeros((batch_size, 3))
        #state[:, :2] = 0.5
        state[:, 0] = 0.5
        # state[:, 1] = 0.2
        state[:, 1] = 0.3

        return state, self.goals

    def step(self, action):
        from tools.utils import clamp
        #print(self.goals.shape, self.state.shape)
        #print(((self.state[0][None, :2] - self.goals)**2).sum(1))
        #exit(0)
        #return super().step(clamp(action, -1, 1))
        return super().step(action)


def main(**kwargs):
    T = 0.01
    trainer = Trainer.parse(
        env=dict(TYPE='TripleCont', reward_weight=1.),
        actor=dict(
            not_func=False,
            a_head=dict(
                TYPE='Normal',
                linear=True,
                squash=False,
                #std_mode='statewise',
                std_mode='fix_no_grad',
                #std_scale=0.05
                std_scale = 0.01
            )
        ), # initial 0.3 
        info_net=dict(action_weight=1.), # TODO: trying to remove action is possible,
        rpg=dict(
            optim=dict(lr=0.0003),
            # use_action_entropy=True,
            weight=dict(reward=1000.0 * T, ent_a=1., ent_z=20., mutual_info=1.) # dwongrade the gradient
        ),
        # backbone=dict(TYPE='PointNet'),
        increasing_reward = 1,

        z_dim=10, z_cont_dim=3,
        #batch_size=8,
        # batch_size=500,
        batch_size=10,
        n_batches=50,
        #max_epochs=30,
        max_epochs = 200,

        _update = kwargs,

        record_gif=True,

        path = 'exp/plot_env/rpg4',
    )
    trainer.epoch_hooks.append(show_curve)
    trainer.start()


#main(rpg=dict(optim=dict(accumulate_grad=5)), batch_size=25, n_batches=250)
#main(max_epochs=100, rpg=dict(weight=dict(reward=40.)))
main(max_epochs=100, rpg=dict(weight=dict(reward=1.)))