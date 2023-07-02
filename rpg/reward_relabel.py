# expedient for relabeling rewards for exploration buffers
import torch
from tools.utils import tensor_like, totensor

def relabel(method, s1, a, s2):
    if method == 'ant2':
        # for AntMaze2 only
        #dtype = gettensortype(s2)
        s2 = totensor(s2, device='cuda:0')
        reached = s2[..., :2] * 100. * 4
        reward = torch.logical_and((torch.abs(reached[..., 0] - 0.5) < 0.5), (torch.abs(reached[..., 1] - 3.5) < 0.5))[..., None]
        return tensor_like(reward, s1)
    else:
        raise NotImplementedError