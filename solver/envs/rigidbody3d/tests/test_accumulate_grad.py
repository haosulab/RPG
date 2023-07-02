import torch
from tools.optim import OptimModule

class SGD(OptimModule):
    def __init__(
        # maxlen is the length of the buffer
        self, network, cfg=None,
        lr=3e-4, max_grad_norm=None, eps=1e-8, loss_weight=None, verbose=True,
        training_iter=1, batch_size=128, maxlen=3000,
        accumulate_grad=0,
    ):
        super().__init__(network, cfg)
        self.optimizer = torch.optim.SGD(self.params, lr=cfg.lr)


value = torch.nn.Parameter(
    torch.tensor(1.0, dtype=torch.float32)
)
lossfn = lambda x: (x ** 2)
opt = SGD(value, lr=1.0, accumulate_grad=2)

for i in range(4):
    print(f"iter:{i}")
    loss = lossfn(value)
    print("before optimize")
    print("\tgrad:", value.grad.item() if value.grad is not None else None)
    print("\tvalue:", value.item())
    opt.optimize(loss)
    print("after optimize")
    print("\tgrad:", value.grad.item())
    print("\tvalue:", value.item())
