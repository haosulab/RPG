import torch
from .density_estimator import DensityEstimator
# import torch_scatter

class Count(DensityEstimator):
    def __init__(self, space, cfg=None, normalizer='none') -> None:
        super().__init__(space, cfg)
        self.buffer = None
        self.discretizer = None 
        
    def register_discretizer(self, discretizer):
        self.discretizer = discretizer

    def make_network(self, space):
        return torch.nn.Linear(1, 1)

    def _log_prob(self, samples):
        assert len(samples.shape) == 2

        index, N = self.discretizer(samples)
        count = torch_scatter.scatter_add(torch.ones_like(index), index, dim=0, dim_size=N)
        if self.buffer is not None:
            self.buffer = self.buffer + count

        prob = (count.float() / count.sum())[:, None]
        return torch.log(prob[index])

    def _update(self, samples):
        #return super()._update(samples)
        index, N = self.discretizer(samples)
        if self.buffer is None:
            self.buffer = torch.zeros(N, device=samples.device, dtype=torch.long)

        log_prob = self._log_prob(samples)

        torch_scatter.scatter_add(torch.ones_like(index, dtype=torch.long), index, out=self.buffer, dim=0)
        return log_prob