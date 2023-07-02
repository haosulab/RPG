# utility function like concat, stack and so on ..
import numpy as np
from torch import nn
from ..operator import Operator
from ..types.tensor import TensorType, Arrow, TupleType

class FlattenBatch(Operator):
    def forward(self, x):
        return x.reshape(-1, *self.inp_types[0].data_shape().as_int())

    def _type_inference(self, inp_type):
        return inp_type.new(inp_type.batch_shape().total(), *inp_type.data_shape())
    
class Flatten(Operator):
    def forward(self, x):
        dims = self.inp_types[0].data_dims
        return x.reshape(*x.shape[:-dims], -1)


    def _type_inference(self, inp_type):
        # print(inp_type.data_shape())
        return inp_type.new(*inp_type.batch_shape(), inp_type.data_shape().total(), data_dims=1)

class Seq(Operator):
    def __init__(self, *modules) -> None:
        self.op_list = modules
        super().__init__()

    def build_modules(self, *args):
        #self.main = Seq(*self._modules)
        pass

    def forward(self, *args, **kwargs):
        out = args
        for i in self.op_list:
            out = [i(*out)]
        return out[0] 

    def _type_inference(self, *args, **kwargs):
        return self.op_list[-1].get_output().get_type()

    def __str__(self) -> str:
        out = 'Sequential:\n'
        for i in self.op_list:
            out += '  ' + str(i).replace('\n', '\n   ') + '\n'
        out += str(self.out)
        return out

        
class Concat(Operator):
    def forward(self, *args):
        import torch
        return torch.cat(args, dim=self.dim)

    def _type_inference(self, *args, **kwargs):
        assert len(args) > 1
        out = args[0]
        for i in args[1:]:
            out = out.new(*out.batch_shape(), *out.data_shape().concat(i.data_shape(), self.dim))
        return out

class Tuple(Operator):
    def forward(self, *args):
        return args

    def get_n_output(self, *input_nodes):
        return len(input_nodes)

    def _type_inference(self, *args):
        return TupleType(*args)


class Linear(Operator):
    @classmethod
    def _new_config(cls):
        return dict(
            dim=256,
        )

    def build_modules(self, inp_type):
        self.myassert(isinstance(inp_type, TensorType), "Linear only support TensorType but got: " + str(inp_type))
        self.myassert(inp_type.data_dims == 1, "Linear only support 1D TensorType but got: " + str(inp_type))
        self.main = nn.Linear(inp_type.channel_dim, self.config.dim).to(inp_type.device)
    
    def _type_inference(self, inp_type):
        return inp_type.new(*inp_type.batch_shape(), self.config.dim)

    def forward(self, *args, **kwargs):
        return self.main(*args, **kwargs)


if __name__ == '__main__':
    pass