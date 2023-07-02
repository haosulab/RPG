import torch
import torch.nn as nn
from tools import dist_utils
from tools.config import Configurable
from .optim import OptimModule

class COptim(Configurable):
    def __init__(
        self,
        network,
        n_constraints,
        cfg=None,
        clip_lmbda=(0.1, 1e10),
        weight_penalty=0.0001,
        max_grad_norm=None,
        reg_proj=0.01,
        constraint_threshold=0.,
        mu=1.,
        lr=0.001
    ) -> None:
        super().__init__()

        self.network = network

        dist_utils.sync_networks(network)
        self.params = list(
            network.parameters() if isinstance(network, nn.Module) else [network])

        self.actions_params = network
        self.loss_optim = torch.optim.Adam(self.params, lr=cfg.lr)
        self.constraint_optim = torch.optim.Adam(self.params, lr=cfg.lr)

        #TODO: change the optim ..
        self.log_alpha = torch.nn.Parameter(
            torch.zeros(n_constraints, requires_grad=True, device=self.params[0].device))
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=cfg.lr)
        self.last_good = None

    def optimize_loss(self, optim, params, loss):
        optim.zero_grad()
        loss.backward()
        dist_utils.sync_grads(params)
        if self._cfg.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(params, self._cfg.max_grad_norm)
        optim.step()

    def optimize(self, loss, C):
        assert loss.shape[0] == C.shape[0]
        n_violate = (C > 0.)
        mask = n_violate.float()


        # penalty 
        penalties = (-torch.log((-C).clamp(min=1e-10)) * self._cfg.weight_penalty * (1-mask))
        loss = (loss * (1-mask))
        #constraints = (C * self.log_alpha.exp() + (C ** 2) * self._cfg.mu/2) * mask
        constraints = (C * self.log_alpha.exp() + (C ** 2) * self._cfg.mu/2) * mask

        #if (C<=0).any():
        #    self.optimize_loss(self.loss_optim, self.params, (loss + penalties).mean())

        self.optimize_loss(self.loss_optim, self.params, (loss + penalties + constraints).mean())

        alpha = self.log_alpha.exp()
        alpha_loss = torch.sum( -self.log_alpha.exp() * C.max().detach()) # c later than 0, increase log_alpha
        self.optimize_loss(self.alpha_optim, [self.log_alpha], alpha_loss)

        return {
            'penalty': float(penalties.mean()),
            'constraint': constraints.mean(),
            'CMax': float(C.max()),
            'CMean': float(C.mean()),
            'alpha': alpha.detach().cpu().numpy(),
            'violation': mask.mean().item()
        }