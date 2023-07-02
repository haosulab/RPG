import copy
import torch
from collections import defaultdict
import numpy as np
from numbers import Number
from typing import List, Union
import numpy as np
import matplotlib

import os
# from tueplots import bundles
# plt.rcParams.update(bundles.neurips2022())

import transforms3d

import random
import torch
import numpy as np



def set_seed(seed: int):
    """
    Args:
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ^^ safe to call this function even if cuda is not available


def enable_full_determinism(seed: int):
    """
    Helper function for reproducible behavior during distributed training. See
    - https://pytorch.org/docs/stable/notes/randomness.html for pytorch
    """
    # set seed first
    set_seed(seed)

    #  Enable PyTorch deterministic mode. This potentially requires either the environment
    #  variable 'CUDA_LAUNCH_BLOCKING' or 'CUBLAS_WORKSPACE_CONFIG' to be set,
    # depending on the CUDA version, so we set them both here
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True)

    # Enable CUDNN deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




class RunningMeanStd(object):
    """Calulates the running mean and std of a data stream.
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    """

    def __init__(
        self, mean: Union[float, np.ndarray] = 0.0, var: Union[float, np.ndarray] = 1.0, clip_max=None, last_dim=True,
    ) -> None:
        self.mean, self.var = mean, var
        self.count = 0
        self.clip_max = clip_max
        self.last_dim = last_dim

        self._submodule = dict()

    def get_submodule(self, k):
        if k not in self._submodule:
            self._submodule[k] = RunningMeanStd(mean=self.mean, var=self.var, clip_max=self.clip_max, last_dim=self.last_dim)
        return self._submodule[k]


    def batch(self, x):
        if isinstance(x, list):
            if isinstance(x[0], np.ndarray):
                x = np.array(x)
            else:
                x = torch.stack(x)
        return x


    @torch.no_grad()
    def update(self, x: np.ndarray) -> None:
        assert isinstance(x, list) or isinstance(x, np.ndarray) or isinstance(x, torch.Tensor), f"{x} {type(x)}"

        if isinstance(x[0], dict):
            # list of dict ..
            for k in x[0]:
                self.get_submodule(k).update([i[k] for i in x])
            return

        x = self.batch(x)

        """Add a batch of item into RMS with the same shape, modify mean/var/count."""
        if self.last_dim and len(x.shape) > 1:
            x = x.reshape(-1, x.shape[-1]) # only keep the last dim, this is used for image and point cloud..
        
        #batch_mean, batch_var = np.mean(x, axis=0), np.var(x, axis=0)
        batch_mean, batch_var = x.mean(axis=0), x.var(axis=0)
        batch_count = len(x)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        new_var = m_2 / total_count

        self.mean, self.var = new_mean, new_var
        self.count = total_count

    
    @property
    def std(self):
        if isinstance(self.var, torch.Tensor):
            return torch.sqrt(self.var).clip(1e-8, np.inf)
        else:
            return np.clip(np.sqrt(self.var), a_min=1E-8, a_max=np.inf)
        
    def normalize(self, x):
        if isinstance(x[0], dict):
            return [{k: self.get_submodule(k).normalize([v])[0] for k, v in c.items()} for c in x]

        x = self.batch(x)
        x = (x - self.mean) / self.std
        if self.clip_max is not None:
            x = x.clip(-self.clip_max, self.clip_max)
        return x

    def unormalize(self, x):
        if isinstance(x[0], dict):
            return [{k: self.get_submodule(k).unormalize([v])[0] for k, v in c.items()} for c in x]
        x = self.batch(x)
        x = x * self.std + self.mean
        return x


def lookat(center, theta, phi, radius):
    R = transforms3d.euler.euler2mat(theta, phi, 0., 'sxyz')
    b = np.array([0, 0, radius], dtype=float)
    back = R[0:3, 0:3].dot(b)
    return R, center - back

def colormap_depth(img):
    if img.shape[-1] == 1:
        img = img[..., 0]

    minima = img.min()
    maxima = img.max()
    norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap='viridis')
    img = np.uint8(mapper.to_rgba(img)[..., :3] * 255)
    return img


def animate(clip, filename='animation.mp4', _return=True, fps=10, embed=False):
    # embed = True for Pycharm, otherwise False
    if isinstance(clip, dict):
        clip = clip['image']
    print(f'animating {filename}')
    if filename.endswith('.gif'):
        import imageio
        import matplotlib.image as mpimg
        imageio.mimsave(filename, clip)
        if _return:
            from IPython.display import display
            import ipywidgets as widgets
            return display(widgets.HTML(f'<img src="{filename}" width="750" align="center">'))
        else:
            return

    from moviepy.editor import ImageSequenceClip
    clip = ImageSequenceClip(clip, fps=fps)
    ftype = filename[-3:]
    if ftype == "mp4":
        clip.write_videofile(filename, fps=fps)
    elif ftype == "gif":
        clip.write_gif(filename, fps=fps)
    else:
        raise NotImplementedError(f"file type {ftype} not supported!")

    if _return:
        from IPython.display import Video
        return Video(filename, embed=embed) 

def read_video(filename, dtype='float32'):
    import imageio
    reader = imageio.get_reader(filename)
    frames = []
    for i, frame in enumerate(reader):
        frames.append(frame)
    if dtype == 'float32':
        frames = [f.astype(float)/255. for f in frames]
    return frames

# operator on dict/list of np array or tensor ..
# the leaf can only be np array or tensor

def dshape(a):
    # print shape or dict of shape 
    if isinstance(a, np.ndarray):
        return "np" + str(tuple(a.shape))
    elif isinstance(a, torch.Tensor):
        return "th" + str(tuple(a.shape))
    elif isinstance(a, list):
        return [dshape(i) for i in a]
    elif isinstance(a, dict):
        return {k: dshape(v) for k, v in a.items()}
    else:
        #raise NotImplementedError
        return a.__class__.__name__

def dstack(a, device):
    if isinstance(a[0], list):
        return batch_input(a, device)
    elif isinstance(a[0], dict):
        return {k: dstack([i[k] for i in a], device) for k, v in a[0].items()}
    else:
        return torch.stack(a, dim=0).to(device)

def dconcat(*args, dim=0):
    if isinstance(args[0], list):
        return [dconcat(row, dim=dim) for row in zip(*args)]
    elif isinstance(args[0], torch.Tensor):
        return torch.cat(args, dim=dim)
    elif isinstance(args[0], np.ndarray):
        return np.concatenate(args, axis=dim)
    elif isinstance(args[0], dict):
        return {k: dconcat(*[i[k] for i in args], dim=dim) for k, v in args[0].items()}
    else:
        raise NotImplementedError

def detach(a):
    # detach tensor or dict of tensor
    if isinstance(a, torch.Tensor):
        return a.detach()
    elif isinstance(a, dict):
        return {k: detach(v) for k, v in a.items()}
    else:
        raise NotImplementedError("Can't detach type {}".format(type(a)))

def dslice(a, slice):
    if isinstance(a, np.ndarray) or isinstance(a, torch.Tensor):
        return a[slice]
    elif isinstance(a, list):
        return [dslice(i, slice) for i in a]
    elif isinstance(a, dict):
        return {k: dslice(v, slice) for k, v in a.items()}
    else:
        raise NotImplementedError

def dmap(a, f):
    if isinstance(a, np.ndarray) or isinstance(a, torch.Tensor):
        return f(a)
    elif isinstance(a, list):
        return [dmap(i, f) for i in a]
    elif isinstance(a, dict):
        return {k: dmap(v, f) for k, v in a.items()}
    else:
        raise NotImplementedError

dmul = lambda a, b: dmap(a, lambda x: x * b)


def batch_input(x, device, dtype=torch.float32):
    if isinstance(x, list):
        if isinstance(x[0], dict):
            x = {k: batch_input([i[k] for i in x], device, dtype) for k in x[0].keys()}
        else:
            if not isinstance(x[0], torch.Tensor):
                x = np.array(x) # directly batch it?
                x = torch.tensor(x, dtype=dtype).to(device)
            else:
                x = torch.stack(x, 0).to(device)
    elif isinstance(x, dict):
        return {k: batch_input(v, device, dtype) for k, v in x.items()}
    elif isinstance(x, np.ndarray):
        return torch.tensor(x, dtype=dtype).to(device)
    elif isinstance(x, torch.Tensor):
        x = x.to(device)
    else:
        raise NotImplementedError("Can't batch type {}".format(type(x)))

    return x

#totensor = batch_input

def gettensortype(x):
    if isinstance(x, torch.Tensor):
        return {'dtype': x.dtype, 'device': x.device, 'cls': 'torch'}
    elif isinstance(x, np.ndarray):
        return {'dtype': x.dtype, 'device': 'cpu', 'cls': 'numpy'}
    else:
        raise NotImplementedError

def totensortype(x, type):
    if type['cls'] == 'torch':
        return torch.tensor(x, dtype=type['dtype']).to(type['device'])
    elif type['cls'] == 'numpy':
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return np.array(x, dtype=type['dtype'])
    else:
        raise NotImplementedError

def tensor_like(a, b):
    return totensortype(a, gettensortype(b))

def tonumpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, list):
        return [tonumpy(i) for i in x]
    elif isinstance(x, tuple):
        return tuple([tonumpy(i) for i in x])
    elif isinstance(x, dict):
        return {k: tonumpy(v) for k, v in x.items()}
    else:
        return x


import time

def timeit(f):

    def timed(*args, **kw):

        ts = time.time()
        result = f(*args, **kw)
        te = time.time()

        print('func:%r args:[%r, %r] took: %2.4f sec' % \
          (f.__name__, args, kw, te-ts))
        return result

    return timed


def info_str(iter_id, **infos):
    #TODO: best loss
    infos = copy.copy(infos)
    word = f'{iter_id}: '
    if 'loss' in infos:
        word = word + f"{infos.pop('loss'):.4f} "
    if 'best_loss' in infos:
        word = word + f"{infos.pop('best_loss'):.4f} "
    if len(infos) > 0:
        word += ', '.join([f'{i}: {infos[i]:.3f}' for i in infos])
    return word

def summarize_info(infos, reduce='sum'):
    assert isinstance(infos, list)
    # setup general reduce function
    def reduce_func(a, b):
        if reduce in {'sum', 'mean'}: 
            return a + b
        elif reduce == 'max':
            return max(a, b)
        elif reduce == 'min':
            return min(a, b)
        else:
            return NotImplementedError
    # summarize info dicts
    outs = defaultdict(float)
    for i in infos: # i: info dict
        for j in i: # j: key
            if j != 'TimeLimit.truncated':
                outs[f"{j}_{reduce}"] = reduce_func(outs[f"{j}_{reduce}"], i[j])
    if reduce == 'mean':
        for i in outs:
            outs[f"{i}_{reduce}"] /= len(infos)
    return outs


def batch_gen(batch_size, *args):
    assert type(batch_size) == int
    length = len(args[0])
    l = 0
    while l < length:
        r = min(l + batch_size, length)
        if len(args) == 1:
            yield args[0][l:r]
        else:
            yield [i[l:r] for i in args]
        l = r


def batched_index_select(input, dim, index):
    assert len(index.shape) <= 2
    views = [input.shape[0]] + \
        [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim, index)


def dict_mean_item(**kwargs):
    return {k: v.mean().item() for k, v in kwargs.items()}


def weighted_sum_dict(kwargs, weights):
    out = None
    info = {}
    for k, v in kwargs.items():
        v = v * float(weights[k])
        if out is None:
            out = v
        else:
            assert out.shape == v.shape, f"Computing weighted sum for {weights}, but out.shape {out.shape} and v.shape {v.shape} do not match!"
            out = out + v
        info[k] = v.mean().item()
    return out, info



def seed(s, env=None):
    if env is not None:
        env.seed(s)
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)


def plt_save_fig_array(fig=None, **kwargs):
    import matplotlib.pyplot as plt
    import io
    if fig is None:
        fig = plt.gcf()

    io_buf = io.BytesIO()
    out = fig.savefig(io_buf, format='raw', **kwargs)#, dpi=DPI) , bbox_inches='tight'
    io_buf.seek(0)
    img_arr = np.frombuffer(io_buf.getvalue(), dtype=np.uint8)
    img_arr = np.reshape(img_arr, newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()
    return img_arr


def clamp(tensor, minval=None, maxval=None):
    if minval is not None:
        # if tensor < minval then return minval.. but the gradient is back-propogate to the tensor ..
        tensor = tensor + torch.relu(minval - tensor).detach()
    if maxval is not None:
        tensor = tensor - torch.relu(tensor - maxval).detach()
    return tensor


def myround(a):
    x = torch.round(a).long()
    assert torch.allclose(x.float(), a), 'the rounding is not correct'
    return x



def orthogonal_init(m):
    from torch import nn
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def ema(m, m_target, tau):
    with torch.no_grad():
        for p, p_target in zip(m.parameters(), m_target.parameters()):
            p_target.data.lerp_(p.data, tau)

class Reshape(torch.nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(*x.shape[:-1], *self.shape)

    def __repr__(self):
        return super().__repr__() + f'(shape={self.shape})'

def mlp(in_dim, mlp_dim, out_dim, act_fn=torch.nn.ELU()):
    """Returns an MLP."""
    if isinstance(mlp_dim, int):
        mlp_dim = [mlp_dim, mlp_dim]
    if isinstance(out_dim, int):
        out_shape = (out_dim,)
    else:
        out_shape = out_dim
        out_dim = int(np.prod(out_dim))
    return torch.nn.Sequential(
        torch.nn.Linear(in_dim, mlp_dim[0]), act_fn,
        torch.nn.Linear(mlp_dim[0], mlp_dim[1]), act_fn,
        torch.nn.Linear(mlp_dim[1], out_dim), Reshape(*out_shape))

class Seq(torch.nn.Module):
    def __init__(self, *main):
        super().__init__()
        self.main = torch.nn.ModuleList(main)
    def forward(self, *args):
        # by default ignore the timestep, or the positional encoding ..
        x = torch.cat([i for i in args if i is not None], dim=-1)
        for m in self.main:
            x = m(x)
        return x


class TimedSeq(torch.nn.Module):
    def __init__(self, *main, positional_embed=False):
        super().__init__()
        self.main = Seq(*main)
        assert not positional_embed
        self.positional_embed = None

    def forward(self, *args, timestep=None):
        assert timestep is not None
        if self.positional_embed:
            args = args + [self.positional_embed(timestep)]
        return self.main(*args)


class CatNet(torch.nn.Module):
    def __init__(self, *args) -> None:
        super().__init__()
        self.main = torch.nn.ModuleList(args)
    def forward(self, *args):
        return torch.cat([m(*args) for m in self.main], dim=-1)

class Identity(torch.nn.Module):
    def forward(self, x):
        return x



def print_input_args(func):
    import inspect
    signature = inspect.signature(func)
    def wrapper(*args, **kwargs):
        print('calling', func.__name__, 'with', {k: dshape(v) for k, v in signature.bind(*args, **kwargs).arguments.items()})
        return func(*args, **kwargs)
    return wrapper