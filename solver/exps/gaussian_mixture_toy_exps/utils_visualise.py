import numpy as np
from toy_reorg import main
from tqdm import tqdm

from matplotlib import pyplot as plt
def run(args, n_run):
    curves = []
    for _ in range(n_run):
        ret = main(args)
        curves.append(ret['rewards'])
    curves = np.array(curves)

    mean = np.mean(curves, axis=0)
    std = np.std(curves, axis=0)
    return mean, std

def plot_curves(args, n_run, POLICYS, COLORS, ax=None, legend_size=8):
    if ax is None:
        fig, ax = plt.subplots(1)
        legend_size = 12
        ax.set_xlabel('step')
        ax.set_ylabel('Reward')
    for policy, color in zip(POLICYS, COLORS):
        print(f'  {policy}, {color}')
        args.policy = policy
        mean, std = run(args, n_run)
        ax.plot(mean, label=policy, color=color)
        ax.fill_between(range(args.max_steps), mean-std, mean+std, alpha=0.2, facecolor=color)
    
    ax.legend(loc='lower right', prop={'size': legend_size})
    ax.yaxis.set_ticks(np.arange(-12.6,6.,2.5))
    ax.set_title(f'bs:{args.batch_size}, grad:{args.grad_size}, maxi:{args.maximum}')
    ax.axhline(y=max(args.maximum,2.0), color='black', linestyle='--')
    #plt.show()

def plot4figs(args, n_run, GRAD_SIZE_LIST, MAXIMUM_LIST, POLICYS, COLORS):
    fig, axs = plt.subplots(len(GRAD_SIZE_LIST), len(MAXIMUM_LIST))
    fig.tight_layout()
    for rid, grad_size in enumerate(GRAD_SIZE_LIST):
        for cid, maximum in enumerate(MAXIMUM_LIST):
            print(f'grad_size: {grad_size}, maximum: {maximum}')
            args.grad_size = grad_size
            args.maximum = maximum
            plot_curves(args, n_run, POLICYS, COLORS, ax=axs[rid, cid])
    fig.align_ylabels(axs)
    plt.savefig(f'{args.init_std}-{args.batch_size}.png')
    '''
    for rid, grad_size in enumerate(GRAD_SIZE_LIST):
        for cid, maximum in enumerate(MAXIMUM_LIST):
            args.grad_size = grad_size
            args.maximum = maximum
            plot_curves(args, n_run, POLICYS, COLORS)
    '''