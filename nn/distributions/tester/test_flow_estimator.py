# experiment for single flow optimization on 2D environments ..  
import torch
import tqdm
import numpy as np
import argparse

from tools.utils import seed, logger
import matplotlib.pyplot as plt

from torch import nn
from pyro import nn as pyro_nn
import pyro.distributions as dist
import pyro.distributions.transforms as T

torch.set_default_dtype(torch.float64)
class ResidualHead(nn.Module):
    def __init__(self, dim, delta=0.1) -> None:
        nn.Module.__init__(self)
        #self.model = pyro_nn.DenseNN(dim, (128, 128))
        self.model = nn.Sequential(
            nn.Linear(dim, 128), 
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, dim),
        )
        self.delta = delta
    
    def forward(self,x ):
        x = self.model(x)
        return torch.tanh(x) * self.delta


def sample_target_density(name, num_sample=100):
    if name == 'two_guassian':
        base_dist = dist.Normal(torch.zeros(2).cuda(), torch.ones(2).cuda())
        mean = torch.tensor([0., 0.5], device='cuda:0')
        std = torch.tensor([0.25, 0.25], device='cuda:0')
        affine = T.AffineTransform(mean, std)
        flow_dist = dist.TransformedDistribution(base_dist, [affine])

        data = flow_dist.sample((num_sample//2,))

        base_dist = dist.Normal(torch.zeros(2).cuda(), torch.ones(2).cuda())
        mean = torch.tensor([-0.4, -0.5], device='cuda:0')
        std = torch.tensor([0.15, 0.1], device='cuda:0')
        affine = T.AffineTransform(mean, std)
        flow_dist = dist.TransformedDistribution(base_dist, [affine])
        data2 = flow_dist.sample((num_sample - num_sample//2,))

        base_dist = dist.Normal(torch.zeros(2).cuda(), torch.ones(2).cuda())
        mean = torch.tensor([1.2, 0.4], device='cuda:0')
        std = torch.tensor([0.15, 0.03], device='cuda:0')
        affine = T.AffineTransform(mean, std)
        flow_dist = dist.TransformedDistribution(base_dist, [affine])
        data3 = flow_dist.sample((num_sample - num_sample//2,))

        return torch.cat((data, data2, data3))
    else:
        raise NotImplementedError


def target_fn(name, num_sample = 100):
    #NOTE: must has a normalized reward range so that the optimization does not suffer.  
    mid0 = torch.tensor([0.5, 0.3, 0.], device='cuda:0')
    mid1 = torch.tensor([-0.5, 0.3, 0.], device='cuda:0')
    def two_mode(x):
        # gradient must be bounded?
        return torch.max(
            torch.exp(-torch.linalg.norm(x - mid0, axis=-1)**2 * 4),
            torch.exp(-torch.linalg.norm(x - mid1, axis=-1)**2 * 3) + 0.05,
        )
        #return (-torch.linalg.norm(x - mid0, axis=-1)**2 * 4).clamp(-3, np.inf)
        #return -torch.linalg.norm(x - mid0, axis=-1)**2 * 4
        flag = (x[..., 0] > 0.).float()
        return (1.-torch.linalg.norm(x - mid0, axis=-1)**2 * 4)*flag +  (1. - torch.linalg.norm(x-mid1, axis=-1)**2 * 4.) * (1-flag)


    return eval(name)



def affine_coupling(input_dim, hidden_dims=None, split_dim=None, dim=-1, **kwargs):
    from functools import partial, reduce
    import operator
    from pyro.nn import DenseNN
    from pyro.distributions.transforms import AffineCoupling
    if not isinstance(input_dim, int):
        if len(input_dim) != -dim:
            raise ValueError(
                "event shape {} must have same length as event_dim {}".format(
                    input_dim, -dim
                )
            )
        event_shape = input_dim
        extra_dims = reduce(operator.mul, event_shape[(dim + 1) :], 1)
    else:
        event_shape = [input_dim]
        extra_dims = 1
    event_shape = list(event_shape)

    if split_dim is None:
        split_dim = event_shape[dim] // 2
    if hidden_dims is None:
        hidden_dims = [10 * event_shape[dim] * extra_dims]

    hypernet = DenseNN(
        split_dim * extra_dims,
        hidden_dims,
        [
            (event_shape[dim] - split_dim) * extra_dims,
            (event_shape[dim] - split_dim) * extra_dims,
        ],
        nonlinearity=torch.nn.Tanh(),
    )
    return AffineCoupling(split_dim, hypernet, dim=dim, **kwargs)


def get_dist_model(name):
    if name == 'FLOW':
        base_dist = dist.Normal(torch.zeros(3).cuda(), torch.ones(3).cuda())
        flows = []
        for i in range(16):
            if False:
                flows.append(T.spline_coupling(2, count_bins=16).cuda())
            else:
                # must be a bijection?
                flows.append(T.affine_coupling(3, hidden_dims=[32, 32]).cuda())
        #for i in range(4): # invertible?
        #    flows.append(T.affine_autoregressive(2).cuda())
        return dist.TransformedDistribution(base_dist, flows), torch.nn.ModuleList(flows).parameters()
    else:
        raise NotImplementedError



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist", type=str, default='two_guassian')
    parser.add_argument("--fn", type=str, default='two_mode')

    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--method", type=str, default='FLOW')

    parser.add_argument("--task", type=str, default='density', choices=['density', 'PG', 'GD', 'PGGD'])

    parser.add_argument("--num_density_sample", type=int, default=1000)

    parser.add_argument("--batch_size", type=int, default=512)

    parser.add_argument("--max_iters", type=int, default=1000)
    parser.add_argument("--render_iters", type=int, default=5)

    parser.add_argument("--lr", type=float, default=0.0002)

    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--residual", default=0, type=int)

    parser.add_argument("--format_strs", type=str, default='csv')

    parser.add_argument("--entropy", type=float, default=0.0)

    args = parser.parse_args()

    logger.configure(args.path, format_strs=args.format_strs)

    if args.batch_size == 1:
        args.lr /= 5.

    if args.seed:
        seed(args.seed)

    if args.task == 'density':
        data = sample_target_density(args.dist, args.num_density_sample).detach() #.cpu().numpy()

        X = data.cpu().numpy()
        plt.scatter(X[:,0], X[:,1], label='data', alpha=0.5)
        logger.savefig('x.png')
    else:
        fn = target_fn(args.fn)
        """
        #from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
        x = np.linspace(-1.5, 1.5, 100)
        X,Y = np.meshgrid(x, x) # grid of point

        

        inp = torch.tensor(np.stack((X, Y), -1), device='cuda:0')

        y = fn(inp).detach().cpu().numpy() # evaluation of the function on the grid
        im = plt.imshow(y,cmap=plt.cm.RdBu) # drawing the function
        plt.colorbar(im) # adding the colobar on the right
        logger.savefig('x.png')
        """



    flow_dist, params = get_dist_model(args.method)
    params = list(params)
    #torch.save(params, 'params'); exit(0)

    with torch.no_grad():
        p = torch.load('params')
        assert len(params) == len(p)
        outs = []
        outs2 = []
        for a, b in zip(params, p):
            assert a.data.shape == b.shape
            a.data[:] = b
            assert torch.allclose(a, b)

            #outs.append(((a.data[:]).abs()).mean().item())
            #outs2.append((b.data[:].abs()).mean().item())
    #print(outs)
    #print(outs2)

    """
    x = np.linspace(-0.99, 0.99, 100)
    X,Y = np.meshgrid(x, x) # grid of point
    inp = torch.tensor(np.stack((X, Y), -1), device='cuda:0').float()
    with torch.no_grad():
        print(flow_dist.log_prob(inp).exp().mean() * 4)
    exit(0)
    """

    if args.residual:
        head = ResidualHead(2).cuda()
        params = list(head.parameters()) + list(params)

        old_fn = fn
        def new_fn(x):
            y = head(x)
            x = x + y
            return old_fn(x)
        fn = new_fn
    else:
        head = None

    optim = torch.optim.Adam(params, lr=args.lr)

    baselines = 0

    for i in tqdm.trange(args.max_iters):
        optim.zero_grad()
        N = 1000000
        for j in tqdm.trange(N):

            if args.task == 'density':

                loss = -flow_dist.log_prob(data.detach()).mean()
                logger.logkv_mean('likelihood', -loss.item())
            else:
                loss = 0.
                samples = flow_dist.rsample((args.batch_size,))

                R = fn(samples)

                baselines = baselines * 0.9 + float(R.mean())*0.1

                logger.logkv_mean('reward', float(R.mean()))

                if args.entropy > 0.:
                    #rand = torch.randn((args.batch_size,2), device=samples.device) * 0.4
                    entropy = -flow_dist.log_prob(samples.detach()).mean()
                    loss += args.entropy * entropy
                    raise NotImplementedError

                R = R/100

            if 'PG' in args.task:
                log_prob = flow_dist.log_prob(samples.detach())
                # log_prob = log_prob.clamp(1e-20, np.inf) # clamp log prob..

                pg = (-log_prob * R.detach()).mean()
                logger.logkv_mean('pg', pg.item())

                loss += pg

            if 'GD' in args.task:
                gd = (-R).mean()

                logger.logkv_mean('gd', gd.item())
                loss += gd

            loss.backward()

            if j % 1000 == 0:
                outs = []
                for i in params:
                    outs.append(((i.grad.data[:]/(j+1)).abs()).mean().item())
                    #outs.append(((i.data[:]).abs()).mean().item())
                print(outs)
        #nn.utils.clip_grad_norm_(params, 0.5)
        #optim.step()
        exit(0)


        if i % args.render_iters == 0:
            X = flow_dist.sample((args.num_density_sample,)).detach().cpu().numpy()
            plt.scatter(X[:,0], X[:,1], label='data', alpha=0.5)
            logger.savefig('y.png')
            logger.dumpkvs()


if __name__ == "__main__":
    main()