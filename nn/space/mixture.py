from ..distributions import ActionDistr, CategoricalAction, DistHead, NormalAction, MixtureAction

class MixtureSpace:
    def __init__(self, discrete, continuous) -> None:
        self.continuous = continuous
        self.discrete = discrete

    def sample(self):
        #return self.continuous.sample()
        import numpy as np
        a = self.discrete.sample()
        b = self.continuous.sample()
        return np.concatenate([[a], b], -1).astype(np.float32)