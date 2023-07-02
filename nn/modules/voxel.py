# networks that may take voxel as input
# we now have probability(x), we want to sample a point cloud according to the distribution.
import torch
from torch import nn
from ..nn_utils import Network

# a sequence 
class VoxelDeconv(Network):
    def __init__(self, inp_dim, cfg=None, start_size=2, mlp_spec=[]) -> None:
        super().__init__()
        self.start_size = start_size
        self.init = nn.Linear(inp_dim, mlp_spec[0] * start_size * start_size * start_size)

        layers = []
        for i in range(len(mlp_spec)-1):
            layers += [
                nn.ConvTranspose3d(mlp_spec[i], mlp_spec[i+1], 3, stride=1, padding=1),
                nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
                nn.LeakyReLU(),
            ]
        self.deconvs = nn.Sequential(*layers)

    def forward(self, x):
        y = self.init(x).reshape(x.shape[0], -1, *[self.start_size] *3)
        return self.deconvs(y)


from .mlp import build_init, ConvModule, _BatchNorm
class VoxelConv(Network):
    def __init__(self, inp_dim, oup_dim, cfg=None, mlp_spec=[256, 256], norm_cfg=None, bias='auto', inactivated_output=True,
                 pretrained=None, conv_init_cfg=dict(type='xavier_init', gain=1, bias=0), norm_init_cfg=None, pooling='avg'):
        super(VoxelConv, self).__init__()
        self.mlp = nn.Sequential()
        mlp_spec = [inp_dim] + mlp_spec + [oup_dim]

        for i in range(len(mlp_spec) - 1):
            if pooling and i:
                if pooling == 'avg':
                    self.mlp.add_module(f'pool{i-1}',nn.AvgPool3d((2, 2, 2)))
                else:
                    self.mlp.add_module(f'pool{i-1}',nn.MaxPool3d((2, 2, 2)))

            if i == len(mlp_spec) - 2 and inactivated_output:
                act_cfg = None
            else:
                act_cfg = dict(type='ReLU')

            self.mlp.add_module(
                f'layer{i}',
                ConvModule(
                    mlp_spec[i],
                    mlp_spec[i + 1],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    dilation=1,
                    groups=1,
                    bias=bias,
                    conv_cfg=dict(type='Conv3d'),
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

                

class ValueOptim(OptimModule):
    def __init__(self, critic, cfg=None,
                 lr=5e-4, vfcoef=0.5):
        super(ValueOptim, self).__init__(critic)
        self.critic = critic
        #self.optim = make_optim(critic.parameters(), lr)
        self.vfcoef = vfcoef

    def compute_loss(self, obs, vtarg):
        vpred = self.critic(obs).mean[..., 0]
        vtarg = batch_input(vtarg, vpred.device)
        assert vpred.shape == vtarg.shape
        vf = self.vfcoef * ((vpred - vtarg) ** 2).mean()
        return vf


class PPO(OptimModule):
    # no need to store distribution, we only need to store actions ..
    def __init__(self,
                 actor,
                 cfg=None,
                 lr=5e-4,
                 clip_param=0.2,
                 max_kl=None,
                 max_grad_norm=None,
                 ):
        super(PPO, self).__init__(actor)
        self.actor = actor
        self.clip_param = clip_param

    def _compute_loss(self, obs, action, logp, adv, backward=True):
        pd: ActionDistr = self.actor(obs)

        newlogp = pd.log_prob(action)
        device = newlogp.device

        adv = batch_input(adv, device)
        logp = batch_input(logp, device)

        # prob ratio for KL / clipping based on a (possibly) recomputed logp
        logratio = newlogp - logp
        ratio = torch.exp(logratio)
        assert newlogp.shape == logp.shape
        assert adv.shape == ratio.shape, f"Adv shape is {adv.shape}, and ratio shape is {ratio.shape}"

        if self.clip_param > 0:
            pg_losses = -adv * ratio
            pg_losses2 = -adv * \
                torch.clamp(ratio, 1.0 - self.clip_param,
                            1.0 + self.clip_param)
            pg_losses = torch.max(pg_losses, pg_losses2)
        else:
            raise NotImplementedError

        assert len(pg_losses.shape) == 2, f"{pg_losses.shape}"
        pg_losses = pg_losses.sum(axis=0).mean()
        #loss = pg_losses

        #approx_kl_div = (ratio - 1 - logratio).mean().item()
        #early_stop = self._cfg.max_kl is not None and (
        #    approx_kl_div > self._cfg.max_kl * 1.5)
        early_stop = False
        assert self._cfg.max_kl is None

        if not early_stop and backward:
            pg_losses.backward()
        else:
            pass

        output = {
            'pg': pg_losses.item(),
            #'approx_kl': approx_kl_div,
        }
        if self._cfg.max_kl:
            output['early_stop'] = early_stop
        return pg_losses, output

