# toy experiments..
# grid to count for 
import matplotlib.pyplot as plt
import torch
import argparse
import numpy as np
import tqdm
from tools.utils import logger
from torch.distributions import Normal

device = 'cuda:0'

def get_fn(name):
    if name == 'TypeA':
        centers = [
                    [0.8],
                    [-0.8],
                ]
        #stds = [0.08, 0.06]
        stds = [0.08, 0.06]
        #heights = [0.21, 0.21, 0.2, 0.23]
        heights = [0.21, 0.3]
    else:
        raise NotImplementedError

    # TODO: we should ensure that the total density of each kernel to be the same ..
    centers = torch.tensor(centers, device='cuda:0')
    heights = torch.tensor(np.array(heights), device='cuda:0')
    stds = torch.tensor(np.array(stds), device='cuda:0')
    def fn(x):
        dist = ((x[..., None, :] - centers[None, :, :]) ** 2).sum(dim=-1)
        r = torch.exp(-dist / stds/stds/2)/stds / 10.  #+ heights
        r = r.max(dim=-1)[0]
        return r
    return fn

def main():
    #TODO: draw function ..
    #TODO: visualize the learning curve ..
    #TODO: measure the success rate ..
    dim = 1

    parser = argparse.ArgumentParser()
    parser.add_argument('--std', default=0.1, type=float)
    parser.add_argument('--learn_std', action='store_true')
    parser.add_argument('--dist', default='TypeA')
    parser.add_argument('--batch_size', default=128, type=int)

    parser.add_argument('--use_rnd', default=20., type=float)
    parser.add_argument('--rnd_decay', default=0.0, type=float)
    parser.add_argument('--ucb', type=int, default=1)

    parser.add_argument('--res', default=32, type=int)
    parser.add_argument('--render_epoch', default=100, type=int)

    parser.add_argument('--seed', default=None, type=int)

    parser.add_argument('--init', default=0., type=float)

    parser.add_argument('--path', default=None, type=str)
    args = parser.parse_args()

    logger.configure(args.path, format_strs='csv')

    

    mu = torch.nn.Parameter(torch.tensor(np.array([args.init]*dim), device=device))
    log_std = torch.nn.Parameter(torch.tensor(np.array([np.log(args.std)]*dim), device=device), requires_grad=args.learn_std)
    params = [mu, log_std]

    reward_fn = get_fn(args.dist)

    x = torch.arange(10000, device='cuda:0')/10000. * 2 - 1
    y = reward_fn(x[:, None])
    plt.plot(x.detach().cpu().numpy(), y.detach().cpu().numpy())
    logger.savefig('fn.png')

    assert args.res > 1

    counter = torch.zeros(args.res, device=device)
    optim = torch.optim.Adam(params, lr=0.01)

    for iter in tqdm.trange(100000):
        optim.zero_grad()


        dist = Normal(mu, log_std.exp())
        a = dist.sample((args.batch_size,))

        logp = dist.log_prob(a.detach()).sum(axis=-1)
        r = reward_fn(a)

        if args.use_rnd > 0.:
            inside_a = torch.logical_and(a >= -1, a <= 1)[:, 0] # bound a

            #a_bin = ((a[inside_a] + 1)/2 * (args.res-1)).long()
            inp = ((a[inside_a, 0] + 1)/2 * (args.res - 1)).long()
            for i in inp:
                counter[i] += 1
            
            counted = torch.bincount(inp)
            assert len(counted) <= len(counter)
            counter[:len(counted)] += counted


            ucb = args.use_rnd * torch.sqrt(np.log((iter+1))/(counter[inp]/args.batch_size+1e-5))
            r[inside_a] += ucb

            #counter[a_bin] += 1
        else:
            ucb = torch.tensor([0.])

        if args.rnd_decay > 0.:
            args.use_rnd = args.use_rnd * (1-args.rnd_decay)

        assert logp.shape == r.shape
        loss = (-logp*r).mean()
        loss.backward()
        optim.step()

        if iter % args.render_epoch == 0:
            print('mean', mu.item(), 'std', log_std.exp().item(), 'r mean', r.mean(), 'ucb mean', ucb.mean())
            logger.logkvs({
                'mean': mu.item(),
                'std': log_std.exp().item(),
                'r_mean': r.mean().item(),
                'ucb_mean': ucb.mean().item(),
            })
            logger.dumpkvs()



if __name__ == '__main__':
    main()