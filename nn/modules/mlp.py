from tools.utils import batch_input
from tools.nn_base import Network
from torch.nn.modules.batchnorm import _BatchNorm
from torch import nn
from .maniskill_modules import ConvModule, build_init, build_norm_layer, build_activation_layer


class ConvMLP(Network):
    def __init__(self, inp_dim, oup_dim, cfg=None, mlp_spec=[256], norm_cfg=None, bias='auto', inactivated_output=True,
                 pretrained=None, conv_init_cfg=dict(type='xavier_init', gain=1, bias=0), norm_init_cfg=None):
        super(ConvMLP, self).__init__()
        self.mlp = nn.Sequential()
        mlp_spec = [inp_dim] + mlp_spec + [oup_dim]

        for i in range(len(mlp_spec) - 1):
            if i == len(mlp_spec) - 2 and inactivated_output:
                act_cfg = None
            else:
                act_cfg = dict(type='ReLU')
            self.mlp.add_module(
                f'layer{i}',
                ConvModule(
                    mlp_spec[i],
                    mlp_spec[i + 1],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    dilation=1,
                    groups=1,
                    bias=bias,
                    conv_cfg=dict(type='Conv1d'),
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=True,
                    with_spectral_norm=False,
                    padding_mode='zeros',
                    order=('conv', 'norm', 'act'))
            )
        self.init_weights(pretrained, conv_init_cfg, norm_init_cfg)

    def forward(self, input):
        return self.mlp(input)

    def init_weights(self, pretrained=None, conv_init_cfg=None, norm_init_cfg=None):
        conv_init = build_init(conv_init_cfg) if conv_init_cfg else None
        norm_init = build_init(norm_init_cfg) if norm_init_cfg else None

        for m in self.modules():
            if isinstance(m, nn.Conv1d) and conv_init:
                conv_init(m)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)) and norm_init:
                norm_init(m)



class LinearMLP(Network):
    def __init__(self, inp_dim, oup_dim, cfg=None, mlp_spec=[], norm_cfg=None, bias='auto', inactivated_output=True, linear_init_cfg=dict(type='xavier_init',gain=1, bias=0), norm_init_cfg=None):
        super(LinearMLP, self).__init__()
        mlp_spec = [inp_dim] + mlp_spec + [oup_dim]

        self.mlp = nn.Sequential()
        for i in range(len(mlp_spec) - 1):
            if i == len(mlp_spec) - 2 and inactivated_output:
                act_cfg = None
                norm_cfg = None
            else:
                act_cfg = dict(type='ReLU')
            bias_i = norm_cfg is None if bias == 'auto' else bias
            # print(mlp_spec[i], mlp_spec[i + 1], bias_i)
            self.mlp.add_module(f'linear{i}', nn.Linear(mlp_spec[i], mlp_spec[i + 1], bias=bias_i))
            if norm_cfg:
                self.mlp.add_module(f'norm{i}', build_norm_layer(norm_cfg, mlp_spec[i + 1])[1])
            if act_cfg:
                self.mlp.add_module(f'act{i}', build_activation_layer(act_cfg))
        self.init_weights(linear_init_cfg, norm_init_cfg)

    def forward(self, input):
        input = input
        return self.mlp(input)

    def init_weights(self, linear_init_cfg=None, norm_init_cfg=None):
        linear_init = build_init(linear_init_cfg) if linear_init_cfg else None
        norm_init = build_init(norm_init_cfg) if norm_init_cfg else None

        for m in self.modules():
            if isinstance(m, nn.Linear) and linear_init:
                linear_init(m)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)) and norm_init:
                norm_init(m)

                


# Resnet Blocks
class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.
    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx