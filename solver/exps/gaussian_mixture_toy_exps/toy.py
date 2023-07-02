"""
Also at different initialization, std, draw performance differences
    and different problems.

batch size also hurts
"""
import torch
import argparse
import tqdm
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
from solver.mixture_of_guassian import GMMAction, NormalAction



def main():
    """
    normal:
        - v1: REINFORCE
        - v2: rsample, and train with gradient
    GMM:
        - v1: REINFORCE + diff gaussian
        - v2: diff all
        - v3: fully reinforce
    Gumbel

    currently:
    - normal-v1, v2
    - GMM-v1, Gumbel
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', default='normal-v1', type=str, choices=[
        'normal-v1', 'normal-v2', 'GMM-v1', 'GMM-v2', 'GMM-v3', 'Gumbel'])
    parser.add_argument('--device', default='cuda:0', type=str)

    parser.add_argument('--entropy', default=0., type=float)
    parser.add_argument('--plot', action='store_true', default=False)
    parser.add_argument('--max_steps', type=int, default=10000)

    parser.add_argument('--grad_size', type=float, default=0.6) # choices 0.6, 10.
    parser.add_argument('--maximum', type=float, default=4.) # choices 4., 0.

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--init_std', type=float, default=0.1)
    args = parser.parse_args()


    def f(x):
        x = torch.clamp(x, -2., 2.)
        assert isinstance(x, torch.Tensor)
        flag = x < 0.5
        return (2.-(x - 0.)**2 * 4)*flag +  (args.maximum - (x-1.)**2 * args.grad_size) * (1-flag.float())

    if args.plot:
        T = 500
        x = torch.arange(T).float().to(args.device)/T * (2 - (-2)) + (-2.)
        y = f(x)
        plt.plot(x.detach().cpu().numpy(), y.detach().cpu().numpy())
        plt.savefig('curve.png')

    if args.policy.startswith('normal'):
        mu = torch.tensor(np.array([0.]), device=args.device)
        log_std = torch.tensor(np.array([np.log(args.init_std)]), device=args.device)
        params = [mu, log_std]
        def action_cls(mu, log_std):
            mu = torch.tanh(mu) * 4 - 2
            return NormalAction(mu, torch.exp(log_std))

    elif args.policy.startswith('GMM') or args.policy == 'Gumbel':
        #raise NotImplementedError
        log_mu = torch.tensor(np.array([0., 0.]), device=args.device)[None, :]
        loc = torch.tensor(np.array([0.2, 0.7]), device=args.device)[None, :]
        log_std = torch.tensor(np.log(np.array([args.init_std, args.init_std])), device=args.device)[None, :]
        params = [log_mu, loc, log_std]
        def action_cls(log_mu, loc, log_std):
            return GMMAction(log_mu, loc, torch.exp(log_std), gumbel=(args.policy == 'Gumbel'))


    params = [nn.Parameter(i) for i in params] 
    optim = torch.optim.Adam(params, lr=0.01)

    losses = []
    rewards = []

    for optim_step in tqdm.trange(args.max_steps):
        optim.zero_grad()

        action = action_cls(*[i for i in params])
        if args.policy == 'normal-v1' or args.policy == 'GMM-v3':
            a = action.sample((args.batch_size,))
        else:
            a = action.rsample((args.batch_size,))

        assert a.shape[:2] == (args.batch_size, 1)

        R = f(a).reshape(-1) # single step ..
        #if optim_step % 10 == 0:
        #    print(params[1].detach().cpu().numpy())

        loss = 0. 
        if args.policy == 'normal-v1' or args.policy == 'GMM-v3':
            #loss += action.REINFORCE(R)
            if args.policy == 'GMM-v3':
                raise NotImplementedError
            log_prob = action.log_prob(a.detach()) 
            assert log_prob.shape == R.shape
            loss -= (log_prob * R.detach()).mean()
            # raise NotImplementedError
        elif args.policy == 'normal-v2' or args.policy == 'GMM-v2' or args.policy == 'Gumbel':
            loss -= R.mean() #minimize -R
        elif args.policy == 'GMM-v1':
            loss -= R.mean()
            loss -= action.REINFORCE(R[:, None]).mean()

        rewards.append(R.detach().cpu().numpy().mean())

        if args.entropy > 0.:
            loss -= args.entropy * action.entropy()

        loss.backward()
        optim.step()

        losses.append(loss.item())


    print(action.mean)
    if isinstance(action, GMMAction):
        print(torch.softmax(action.log_mu, -1))
    print('sample action', action.sample((10,)))
    plt.clf()

    plt.plot(losses, label='loss')
    plt.savefig('loss.png')

    plt.clf()

    plt.plot(rewards, label='reward')
    plt.savefig('reward.png')



if __name__ == '__main__':
    main()