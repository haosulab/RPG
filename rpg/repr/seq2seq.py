from tools.nn_base import Network

class SeqDecoder(Network):

    def __init__(self, state_dim, z_dim, a_dim, cfg=None,
                 T=32, out_dim=4):
        super().__init__()
    
    def forward(self, inputs):
        # input (B, state_dim + z_dim + action_dim)
        # output (T, B, out_dim)
        raise NotImplementedError


class SeqModel(Network):
    """
    we have a trajectory predict model:
    - low-level model predicts s_1, s_2, s_3 .. = tau_{:h}
    - I(s, z) -> {z_t}_{t>=1}

    Maximize info p(z|{z_t}_{t>=1})

    The reward is assigned to all states of the trajectory equally.

    In the other sense, it can be viewed as a trajectory-based mutual information prediction model:
    p(z|I(s, a, z))  

    where I(s, a, z) satisfies the following consistency requirement (TD-learning):
        - z_1 should match tau_{:h}; either z_1 can be predicted by tau_{:h}; or we can reconstruct $z_1$ from tau_{:h}:
            kind of encoder decoder
        - I(tau_{h}, z) matches z_{1:}


    Hidden Traj Model:
    - encoder: tau_{:h} -> z_1
    - decoder: z_1 -> tau_{:h}
    - mutual information predictor: p(z|I(s, a, z))

    where we need it:
    - when model rollout, return z ..
    """
    def __init__(self, state_dim, z_dim, a_dim, cfg=None):
        super().__init__()
        self.state_dim = state_dim
        self.z_dim = z_dim
        self.a_dim = a_dim

        self.I = SeqDecoder(state_dim, z_dim, a_dim)

    def intrinsic_reward(self, states, a_seq, z):
        # given a short horizon states, a_seq and return the 
        return 

    def update(self, states, z, actions):
        """
        state (T+1, B, state_dim)
        z (T+1, B, z_dim)
        actions (T, B, action_dim)

        update two parts
        """
        raise NotImplementedError