# intrinsic motivation
import torch

class IntrinsicMotivation:
    def __init__(self, *args) -> None:
        self.args = args

    def __call__(self, rollout):
        outs = {}
        reward = rollout['reward']

        for i in self.args:
            name, intrinsic = i.intrinsic_reward(rollout)
            if name is None:
                continue
            assert reward.shape == intrinsic.shape
            outs[name] = intrinsic
        return reward, outs

    # def update_with_rollout(self, rollout):
    #     for i in self.args:
    #         i.update_intrinsic(rollout)

    # def update_with_buffer(self, buffer):
    #     for i in self.args:
    #         i.update_with_buffer(buffer)
    
    # def visualize_transition(self, transitions):
    #     # add information for visualization
    #     for i in self.args:
    #         if hasattr(i, 'visualize_transition'):
    #             i.visualize_transition(transitions)