import typing
import numpy as np
import torch
from ..basetypes import Type, VariableArgs, TupleType, Arrow
from ..operator import Operator
from .size import SizeType, UIntType
from torch import nn


class TensorType(Type):
    def __init__(self, *size, data_dims=1, dtype=None, device=None):
        size = SizeType(*size)
        assert isinstance(size, SizeType) # size could be either an size type or a type variable ..
        self.size = size
        self.data_cls = torch.Tensor

        self.dtype = dtype or torch.float32
        self.device = device or 'cuda:0'

        self.data_dims = data_dims
        self._data_dims = UIntType(data_dims)

    def _get_extra_info(self):
        return {
            'dtype': self.dtype,
            'device': self.device,
        }

    def reinit(self, *children):
        return self.__class__(*children[:-1], data_dims=int(children[-1]), **self._get_extra_info())

    def new(self, *size, data_dims=None):
        return TensorType(*size, dtype=self.dtype, device=self.device, data_dims=data_dims or self.data_dims)

    def batch_shape(self):
        return self.size[:-self.data_dims]

    def data_shape(self):
        return self.size[-self.data_dims:]

    @property
    def channel_dim(self):
        C = self.size[-self.data_dims]
        try:
            C = int(C)
        except TypeError as e:
            raise TypeError(str(e) + f"\n    The actual channel is Type {C}")
        return C

    def instance(self, value):
        if not (isinstance(value, self.data_cls)):
            return False
        return self.size.instance(value.shape)

    def children(self):
        return list(self.size.children()) + [self._data_dims]

    def __str__(self):
        #out = self.type_name
        out = '(' + ','.join(map(str, self.batch_shape())) + ' : ' + ','.join(map(str, self.data_shape())) + ')'
        if self.data_dims != 1:
            out = f'{self.data_dims}D' + out
        out = 'Tensor' + out
        if self.dtype != torch.float32:
            out = 'D' + out
        if self.device != 'cuda:0':
            out = 'Cpu' + out
        return out

    def sample(self):
        return torch.randn(*self.size.sample(), device=self.device, dtype=self.dtype)


Tensor1D = TensorType('...', 'N', data_dims=1)

class MLP(Operator):
    INFER_SHAPE_BY_FORWARD=True

    @classmethod
    def _new_config(cls):
        return dict(
            layer=3,
            hidden=512,
            out_dim=32,
            act_fn=None,
        )

    def build_modules(self, inp_type):
        assert inp_type.data_dims == 1, "MLP only support 1D data"
        C = inp_type.channel_dim
        act_fn = self.config['act_fn'] or nn.ELU()
        hidden_dim = self.config.hidden

        self.main = torch.nn.Sequential(
            torch.nn.Linear(C, hidden_dim), act_fn,
            *sum([[torch.nn.Linear(hidden_dim, hidden_dim), act_fn] 
                  for _ in range(self.config.layer-2)], []),
            torch.nn.Linear(hidden_dim, self.config.out_dim)).to(inp_type.device)
        

def test():
    N = Type('N')
    M = Type('M')
    shapeA = TensorType(3, N, M,)
    shapeB = TensorType(VariableArgs('...'), 4, 5)
    
    shapeC = TensorType(N, M, VariableArgs('...'))
    arrow = Arrow(shapeA, shapeB, shapeC)
    print(arrow)

    X = TensorType(3, 4, 5)
    B = TensorType(10, 10, 4, 5)

    arrow.test_unify("Tensor(4,5,10 : 10)", X, B)


    shapeA = TensorType(VariableArgs('...'), 4, 5)
    shapeB = TensorType(N, M, VariableArgs('...'))
    shapeC = TensorType(N, M, VariableArgs('...'))
    arrow = Arrow(shapeA, shapeB, shapeC)

    #print(arrow.unify(X, B))
    arrow.test_unify("error", X, B)

    arrow.test_unify("Tensor(4,5,4 : 5)", TensorType(4, 5, 4, 5), TensorType(4, 5, VariableArgs('...')))

    arrow.test_unify("Tensor(2,2 : 1)", TensorType(1, 4, 5), TensorType(2, 2, 1))

    # arrow.test_unify("error", TensorType(1, 4, 5), TensorType(2, 2, 2))
        
def test_mlp():
    inp = TensorType(3, 4, 5, data_dims=1)

    mlp = MLP(inp, layer=5, hidden=512, out_dim=32)
    print(mlp)
    mlp(inp.sample())
        
if __name__ == '__main__':
    test()
    test_mlp()