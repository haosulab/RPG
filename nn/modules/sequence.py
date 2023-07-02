import torch
from .attention import BasicTransformerBlock
from tools.utils import mlp
from tools.nn_base import Network


class SequentialBackbone(Network):
    def __init__(self, inp_dim, context_dim, output_dim, cfg=None, dim=256, length=1, depth=3, n_heads=1, layer_norm=False) -> None:
        super().__init__()
        self.inp_dim = inp_dim
        self.context_dim = context_dim
        self.output_dim = output_dim
        self.length = length

        self.positional_encoding = torch.nn.Embedding(length + 1, dim)
        self.encoder = mlp(inp_dim, dim, (length, dim))
        self.context_proj = torch.nn.Linear(context_dim, dim)
        self.out = mlp(dim, dim, output_dim)

        d_head = dim//n_heads

        self.transformer_blocks = torch.nn.Sequential(
            *[BasicTransformerBlock(dim, n_heads, d_head, dropout=False, context_dim=None, layer_norm=layer_norm)
                for d in range(depth)]
        )

        self.auxilary = mlp(inp_dim, dim, (context_dim, output_dim))

    def forward(self, x: torch.Tensor, context: torch.Tensor):
        assert x.shape[-1] == self.inp_dim
        assert context.shape[-1] == self.context_dim


        shape = x.shape[:-1]

        x = x.reshape(-1, self.inp_dim)
        context = context.reshape(-1, self.context_dim)

        # out = (self.auxilary(x) * context[:, :, None]).sum(dim=1)

        seq = self.encoder(x)
        context = self.context_proj(context)[:, None, :]

        index = torch.arange(self.length + 1, device=self.positional_encoding.weight.device)
        inp = torch.concat([context, seq], dim=1) + self.positional_encoding(index)[None, :, :] # make it not too big

        #out = self.transformer_blocks(inp)[:, 0]
        out = self.transformer_blocks(inp).mean(axis=1)
        # out = (seq + context)[:, 0]
        #out = self.out(out)/100.
        out = self.out(out)
        assert out.shape[-1] == self.output_dim
        return out.reshape(*shape, self.output_dim) #initialize to zero ..