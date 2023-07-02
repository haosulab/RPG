from torch import nn
import termcolor

import torch
from ..operator import Operator
from .tensor import TensorType, VariableArgs, Type
from ..node import Node
from ..functors import Flatten, Seq, Linear, FlattenBatch, Tuple

ImageType = TensorType('...', 'D', 'N', 'M', data_dims=3)


class ConvNet(Operator):
    INFER_SHAPE_BY_FORWARD=True

    @classmethod
    def _new_config(cls):
        return dict(
            layer=4,
            hidden=512,
            out_dim=32,
        )

    def build_modules(self, inp_type):
        try:
            int(inp_type.data_shape().total())
        except TypeError:
            raise TypeError(f'ConvNet requires a fixed input shape but receives {str(inp_type)}')
        assert inp_type.data_dims is 3
        C = inp_type.channel_dim

        num_channels = self.config.hidden
        self.main = nn.Sequential(
            nn.Conv2d(C, num_channels, 7, stride=2), nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, 5, stride=2), nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, 3, stride=2), nn.ReLU(),
            *sum([[nn.Conv2d(num_channels, num_channels, 3, 2), nn.ReLU()] for _ in range(max(self.config.layer-4, 0))], []),
            nn.Conv2d(num_channels, self.config.out_dim, 3, stride=2), nn.ReLU()
        ).to(inp_type.device)


def test_conv():
    inp = TensorType('N', 'M', 5,224,224, data_dims=3)

    # assert inp.instance(torch.zeros([5, 5, 224, 224]))
    # assert inp.instance(torch.zeros([5, 224, 224]))

    flattenb = FlattenBatch(inp)
    
    conv = ConvNet(flattenb, layer=4)
    flatten = Flatten(conv)
    
    linear = Linear(flatten, dim=20)

    linear3, other = Tuple(linear, flatten)
    linear2 = Linear(linear3, dim=10)

    out = Tuple(linear, linear2)
    graph = out.compile(config=dict(Linear=dict(dim=35)))
    print(graph.pretty_config)

    seq = Seq(flattenb, conv, flatten, linear, linear2)

    image = inp.sample()
    from omegaconf import OmegaConf as C
    
    assert torch.allclose(graph(image)[1], seq(image))


    img = torch.tensor([1., 2., 3.])
    try:
        graph(img)
    except TypeError as e:
        print(termcolor.colored(str(e), 'red'))
    print("OK!")

    print('conv parameters', len(list(conv.parameters())))
    print('graph parameters', len(list(graph.parameters())))
    


if __name__ == '__main__':
    test_conv()