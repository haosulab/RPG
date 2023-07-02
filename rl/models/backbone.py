# base network..
import numpy as np
import torch
from gym.spaces import Space
from typing import Optional
from torch import nn
from .utils import PointNetSetAbstraction
from .utils import get_graph_feature
from tools.config import Configurable, as_builder
from tools.utils import batch_input


ACTIVATIONS = {
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
}



@as_builder
class Backbone(Configurable, nn.Module):
    # batchbone will track the device
    def __init__(
        self,
        obs_space: Space,
        action_space: Optional[Space],
        cfg=None
    ):
        super(Backbone, self).__init__(cfg)
        nn.Module.__init__(self)
        self.obs_space = obs_space
        self.action_space = action_space

        self._device = None
    
    def batch_input(self, x):
        return batch_input(x, self.device)

    def to(self, device):
        self._device = device
        return nn.Module.to(self, device)

    def cuda(self, device='cuda:0'):
        self._device = device
        return nn.Module.cuda(self, device)

    @property
    def device(self):
        #return next(self.parameters()).device
        if self._device is not None:
            return self._device
        else:
            self._device = next(self.parameters()).device
            return self._device

    @property
    def output_shape(self):
        return self._output_shape


class MLP(Backbone):
    def __init__(
        self,
        obs_space,
        action_space,
        cfg=None,

        dims=(256, 256),
        layers=None,
        dim_per_layer=256,
        output_relu=True,
        activation='relu'
    ):
        Backbone.__init__(self, obs_space, action_space, cfg=cfg)

        if layers is not None:
            dims = (dim_per_layer,) * layers

        input_shape = list(obs_space.shape)
        assert len(obs_space.shape) == 1
        self.x_dim = obs_space.shape[0]

        if action_space is not None:
            input_shape[0] += action_space.shape[0]
            assert len(action_space.shape) == 1
            self.action_dim = action_space.shape[0]

        dims = (int(np.prod(input_shape)),) + dims

        self.dims = dims
        self.input_shape = input_shape
        assert len(dims) >= 2

        layers = []

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            if i != len(dims) - 2 or output_relu:
                layers.append(ACTIVATIONS[activation]())

        self.main = nn.Sequential(*layers)
        self._output_shape = (dims[-1],)

    def forward(self, x, a=None):
        x = batch_input(x, self.device)
        x_shape = x.shape
        x = x.reshape(*x_shape[:-1], self.x_dim)

        if a is not None:
            a = batch_input(a, self.device)
            a = a.reshape(*a.shape[:-1], self.action_dim)
            x = torch.cat((x, a), -1)

        assert x.shape[-1] == self.dims[0], f"{x.shape[1], self.dims[0]}"
        output = self.main(x)
        return output



class ConvNet(Backbone):
    # MiniCNN: 3-layer conv + 1-layer FC
    def __init__(self, obs_space, action_space, cfg=None,
                 dims=(32, 64, 64, 128),
                 kernels=(8, 4, 3),
                 strides=(4, 2, 1),
                 padding=0,
                 output_relu=True, activation='relu',
                 divide255=True):
        Backbone.__init__(self, obs_space, None, cfg=cfg)

        self.divide255 = divide255

        input_shape = obs_space.shape
        dims = (input_shape[0] + (action_space.shape[0] if action_space is not None else 0),) + dims
        self.input_shape = (dims[0], *input_shape[1:])
        assert len(dims) >= 2

        layers = []
        for i in range(len(dims) - 2):  # 2 layer x 3 Conv-stack
            layers.append(nn.Conv2d(dims[i], dims[i + 1], kernel_size=kernels[i], stride=strides[i], padding=padding))
            #layers.append(nn.Dropout2d(0.5)) # augmentation - Dropout2D
            layers.append(ACTIVATIONS[activation]())
        self.conv = nn.Sequential(*layers)
        layers = [nn.Linear(self.feature_size(), dims[-1])] # 2 layer x 1 FC-stack
        #layers.append(nn.Dropout(0.5)) # augmentation - Dropout1D
        if output_relu:
            layers.append(ACTIVATIONS[activation]())
        self.feature = nn.Sequential(*layers)
        self._output_shape = (dims[-1],)

    def feature_size(self):
        return self.conv(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)

    def forward(self, x, a=None):  # x.shape: [B, 3x2, 128, 128]
        if self.divide255:
            x = x.float()/255.
        #print(x.shape)
        #assert isinstance(x, torch.FloatTensor), f"{x.dtype}, {type(x)}"

        assert x.dim() == 4, f"Input shape ({x.size()}) is not image!"
        if a is not None:
            if a.dim() < x.dim():
                a = a[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
            x = torch.cat((x, a), dim=1)
        x = self.conv(x) # x.shape: [B, 64, 12, 12]
        x = x.reshape(x.size(0), -1)
        x = self.feature(x) # x.shape: [B, 128]
        return x
    
    
    
class PointNet(Backbone):
    # MiniPointNet: 3-layer MLP (w/o BatchNorm) + MaxPool
    def __init__(self, obs_space, action_space, cfg=None,
                 dims=(32, 64, 128),
                 output_relu=True, activation='relu'):
        Backbone.__init__(self, obs_space, None, cfg=cfg)

        input_shape = obs_space.shape  # (Cin, 1024) 
        dims = (input_shape[0] + (action_space.shape[0] if action_space is not None else 0),) + dims # [Cin+a, 32, 64, 128]
        self.input_shape = (dims[0], *input_shape[1:]) # [Cin+a, 1024]
        assert len(dims) >= 2
        
        layers = []
        for i in range(1, len(dims)): # 3 layer x 3 MLP-stack
            layers.append(nn.Conv1d(dims[i-1], dims[i], kernel_size=1, bias=False))
            layers.append(ACTIVATIONS[activation]())        
        if not output_relu:
            del layers[-1]
        self.conv = nn.Sequential(*layers)
        self._output_shape = (dims[-1],)

    def forward(self, x, a=None): # x.shape: [B, Cin, 1024]
        assert x.dim() == 3, f"Input shape ({x.size()}) is not pointcloud!"
        if a is not None:
            #TODO: untested for pointcloud data with action given (assume a is of shape [?, 1024])
            if a.dim() < x.dim():
                a = a[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
            x = torch.cat((x, a), dim=1)
        x = self.conv(x).max(dim=-1, keepdim=False)[0]  # x.shape: [B, 128]   
        return x
    
class DGCNN(Backbone):
    # MiniDGCNN: 3 EdgeGCN (w/o BatchNorm) + Concat
    def __init__(self, obs_space, action_space, cfg=None,
                 dims=(32, 64, 128),
                 output_relu=True, activation='relu'):
        Backbone.__init__(self, obs_space, None, cfg=cfg)

        input_shape = obs_space.shape  # (Cin, 1024) 
        dims = (input_shape[0] + (action_space.shape[0] if action_space is not None else 0),) + dims # [Cin+a, 32, 64, 128]
        self.input_shape = (dims[0], *input_shape[1:]) # [Cin+a, 1024]
        self._output_shape = (dims[-1],)
        assert len(dims) >= 2
        assert output_relu, 'w/o-last-relu version is not implemented for DGCNN yet.'

        self.k = 20
        self.conv1 = nn.Sequential(nn.Conv2d(dims[0] * 2, dims[1], kernel_size=1, bias=False), ACTIVATIONS[activation]())
        self.conv2 = nn.Sequential(nn.Conv2d(dims[1] * 2, dims[2], kernel_size=1, bias=False), ACTIVATIONS[activation]())
        self.conv3 = nn.Sequential(nn.Conv2d(dims[2] * 2, dims[3], kernel_size=1, bias=False), ACTIVATIONS[activation]())
        self.conv_final = nn.Sequential(nn.Conv1d(dims[1] + dims[2] + dims[3], dims[-1], kernel_size=1, bias=False), ACTIVATIONS[activation]())
        
    def forward(self, x, a=None): # x.shape: [B, Cin, 1024]
        assert x.dim() == 3, f"Input shape ({x.size()}) is not pointcloud!"
        if a is not None:
            #TODO: untested for pointcloud data with action given (assume a is of shape [?, 1024])
            if a.dim() < x.dim():
                a = a[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
            x = torch.cat((x, a), dim=1)
            

        x1 = self.conv1(get_graph_feature(x, k=self.k)).max(dim=-1, keepdim=False)[0]
        x2 = self.conv2(get_graph_feature(x1, k=self.k)).max(dim=-1, keepdim=False)[0]
        x3 = self.conv3( get_graph_feature(x2, k=self.k)).max(dim=-1, keepdim=False)[0]
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.conv_final(x).max(dim=-1, keepdim=False)[0]  # x.shape: [B, 128]   
        return x    
    
    
class PointNet2(Backbone):
    # MiniPointNet++(SSG): 2 1-MLP SA-stack (w/o BatchNorm)
    def __init__(self, obs_space, action_space, cfg=None,
                 dims=(32, 64, 128),
                 output_relu=True, activation='relu'):
        Backbone.__init__(self, obs_space, None, cfg=cfg)

        input_shape = obs_space.shape  # (Cin, 1024) 
        dims = (input_shape[0] + (action_space.shape[0] if action_space is not None else 0),) + dims # [Cin+a, 128, 256, 128]
        self.input_shape = (dims[0], *input_shape[1:]) # [Cin+a, 1024]
        self._output_shape = (dims[-1],)
        assert len(dims) >= 2
        assert output_relu, 'w/o-last-relu version is not implemented for PointNet2 yet.'
        
        # 3 3-MLP SA-stack (20mins per step; 8GB) (Cin -> 32 -> 64 -> 128)
        #self.sa1 = PointNetSetAbstraction(npoint=input_shape[1]//2, radius=0.2, nsample=32, in_channel=dims[0], mlp=[dims[1]//2, dims[1]//2, dims[1]], group_all=False)
        #self.sa2 = PointNetSetAbstraction(npoint=input_shape[1]//8, radius=0.4, nsample=64, in_channel=dims[1] + 3, mlp=[dims[1], dims[1], dims[2]], group_all=False)
        #self.sa3 = PointNetSetAbstraction(npoint=None,           radius=None, nsample=None, in_channel=dims[2] + 3, mlp=[dims[2], dims[2], dims[-1]], group_all=True) 
        
        # 3 2-MLP SA-stack (16mins per step; 8GB) (Cin -> 32 -> 64 -> 128)
        #self.sa1 = PointNetSetAbstraction(npoint=input_shape[1]//2, radius=0.2, nsample=32, in_channel=dims[0], mlp=[dims[1]//2, dims[1]], group_all=False)
        #self.sa2 = PointNetSetAbstraction(npoint=input_shape[1]//8, radius=0.4, nsample=64, in_channel=dims[1] + 3, mlp=[dims[1], dims[2]], group_all=False)
        #self.sa3 = PointNetSetAbstraction(npoint=None,           radius=None, nsample=None, in_channel=dims[2] + 3, mlp=[dims[2], dims[-1]], group_all=True) 

        # 3 1-MLP SA-stack (14mins per step; 8GB) (Cin -> 32 -> 64 -> 128)
        #self.sa1 = PointNetSetAbstraction(npoint=input_shape[1]//2, radius=0.2, nsample=32, in_channel=dims[0], mlp=[dims[1]], group_all=False)
        #self.sa2 = PointNetSetAbstraction(npoint=input_shape[1]//8, radius=0.4, nsample=64, in_channel=dims[1] + 3, mlp=[dims[2]], group_all=False)
        #self.sa3 = PointNetSetAbstraction(npoint=None,           radius=None, nsample=None, in_channel=dims[2] + 3, mlp=[dims[-1]], group_all=True) 
        
        # 2 1-MLP SA-stack (14mins per step; 6GB) (Cin -> 64 -> 128)
        self.sa1 = PointNetSetAbstraction(npoint=input_shape[1]//4, radius=0.2, nsample=32, in_channel=dims[0], mlp=[dims[2]], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=None,           radius=None, nsample=None, in_channel=dims[2] + 3, mlp=[dims[-1]], group_all=True) 

    def forward(self, x, a=None): # x.shape: [B, Cin, 1024]
        assert x.dim() == 3, f"Input shape ({x.size()}) is not pointcloud!"
        if a is not None:
            #TODO: untested for pointcloud data with action given (assume a is of shape [?, 1024])
            if a.dim() < x.dim():
                a = a[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
            x = torch.cat((x, a), dim=1)
        
        xyz, feats = x[:, :3, :], x[:, 3:, :]
        B, _, _ = xyz.shape
        l1_xyz, l1_feats = self.sa1(xyz, feats)
        l2_xyz, l2_feats = self.sa2(l1_xyz, l1_feats)
        #l3_xyz, l3_feats = self.sa3(l2_xyz, l2_feats)
        #x = l3_feats.view(B, 128) # x.shape: [B, 128]  
        
        x = l2_feats.view(B, 128) # x.shape: [B, 128]   
        return x