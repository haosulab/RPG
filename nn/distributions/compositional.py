import torch
from .dist_head import ActionDistr

def where(a, b, c):
    if isinstance(b, tuple):
        assert len(b) == 2
        return where(a, b[0], c[0]), where(a, b[1], c[1])

    while a.dim() < b.dim():
        a = a[..., None]
    a = a.expand_as(b)
    return torch.where(a, b, c)

class Compose(ActionDistr):
    def __init__(self, A, B, mask) -> None:
        self.A = A
        self.B = B
        self.mask = mask

    def rsample(self):
        return where(self.mask, self.A.rsample(), self.B.rsample())

    def sample(self):
        return where(self.mask, self.A.sample(), self.B.sample())

    def log_prob(self, action):
        return where(self.mask, self.A.log_prob(action), self.B.log_prob(action))

    def entropy(self):
        return where(self.mask, self.A.entropy(), self.B.entropy())
