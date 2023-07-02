import numpy  as np
from rpg.scripts import plot
import matplotlib.pyplot as plt


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    #plot_ablation(args.runs, args.keys)

    tasks = [
        ['Latent space', ['gaussian', 'discrete', 'mbsac'], ['Gaussian', 'Discrete', 'No Latent Space']],
        ['beta', ['gaussian', 'gaussian0', 'gaussian0001', 'gaussian005', 'gaussian05'], ['0.01', '0.0', '0.0001', '0.005', '0.05']],
        ['Latent space dimension', ['gaussian', 'gaussiand6', 'gaussiand3', 'gaussiand1'], ['d=12', 'd=6', 'd=3', 'd=1']],
        ['RND', ['gaussian', 'rndnobuf', 'rndd0'], ['Ours', 'No buffer', 'No emmbedding']],
        ['Policy parameterization', ['gaussian', 'mbsac', 'gmm', 'flow', 'mpc'], ['Ours', 'Gaussian', 'GMM', 'Flow', 'CEM']],
    ]

    width = min(10, len(tasks))
    n_rows = (len(tasks) + width - 1)//width
    fig, axs = plt.subplots(n_rows, width, figsize=(8 * width, 8 * n_rows))
    
    if isinstance(axs, np.ndarray):
        if isinstance(axs[0], np.ndarray):
            axs = sum([list(x) for x in axs], [])
    else:
        axs = [axs]

    def plot_ablation(idx, name, runs, keys=None):
        ax = axs[idx]
        if keys is None:
            keys = runs

        color_id = 0
        plot.KEYS['gapexplore'] = 'train_occ_metric'
        X = []
        Y = []
        COLORS = ['C3', 'C0', 'C1', 'C2', 'C4']

        ax.plot([-10., 1000], [0., 0.], linewidth=2, color='black')
        ax.plot([0., 0.], [-1., 2.], linewidth=2, color='black')

        for k, run in zip(keys, runs):
            v = plot.read_wandb_result('gapexplore', run)

            if len(v[0]) > 0:
                print(k, len(v[0]), [len(x) for x in v[0]])
                x, mean, std = plot.merge_curves(v[0], v[1], 10000, None)
                x = np.append(0, x)
                mean = np.append(1./25, mean) # 
                std = np.append(0, std)
                kwargs = {}
                if k == 'Flow':
                    x = np.append(x, 0.24625)
                    std = np.append(std, 0) 
                    mean = np.append(mean, mean[-1]) 
                    kwargs['linestyle'] = '--'
                plot.plot_curve_with_shade(ax, x, mean, std * 2.,
                                    label=k, smoothingWeight=0.05, color = COLORS[color_id], linewidth=4, **kwargs)

                # X.append(x)
                # Y.append(mean[-1])
                
        #ax.bar(X, Y)
            color_id += 1

        ax.legend(loc=2)
        ax.set_title("("+chr(ord('A') + idx) +") " + name)
        ax.set_xlabel("interactions (M)"); 
        ax.set_ylabel('state coverage')
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(-0.01, 0.24)
        ax.grid()

    for i, task in enumerate(tasks):
        plot_ablation(i, *task)

    plt.tight_layout()
    plt.savefig('ablation.png', dpi=300)