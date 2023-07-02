import torch
import matplotlib.pyplot as plt
from rpg.scripts.plot import create_axes
import numpy as np
from solver.draw_utils import plot_colored_embedding

def group(images):
    method = list(images.keys())
    n_figs = len(images[method[0]])

    outs = []
    for i in images:
        outs += list(images[i])

    axs = create_axes(n_figs, outs)
    titles = [
        '(A) Epoch 1', '(B) Epoch 20', '(C) Visitation Count',
        '', '', ''
    ]
    ytitle = [
        'Gaussian', '', '', 'Our Method', '', ''
    ]
    for img_path, ax, t, y in zip(outs, axs, titles, ytitle):
        if isinstance(img_path, str):
            img = plt.imread(img_path)
            if 'occ' not in img_path:
                x = 280
                img = img[x:-x, x:-x, :3]
        else:
            img = img_path
        ax.imshow(img)
        ax.set_title(t)
        ax.set_ylabel(y, fontdict=dict(fontsize=50))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_ticks([])

        # ax.axis('off')
        


def plot_bandit():
    images = dict(
        sac=['Reinforce_000', 'Reinforce_003', 'Reinforce_010', 'Reinforce_022'],
        ours=['RPG_000', 'RPG_010', 'RPG_011', 'RPG_032'],
    )

    for i in images:
        #images[i] = [torch.load(f'data/savedbuffer/{i}/buffer.pt') for i in images[i]]
        images[i] = [f'data/plots/bandit/{j}.png' for j in images[i]]

    group(images)
    plt.tight_layout()
    plt.savefig(f'data/bandit.png', dpi=300)


def plot_maze():
    plt.figure(figsize=(10, 10))
    import pandas as pd
    X = pd.read_csv(f'data/buffers/mbsac_seed6/progress.csv')['train_occ_metric'].dropna()
    X.index //= 50
    Y = pd.read_csv(f'data/buffers/gaussian_seed4/progress.csv')['train_occ_metric'].dropna()
    Y.index //= 50

    import scipy
    def smooth(X):
        x = X.index[::5] + 1
        x = np.append(0, x)
        ys = [0.1]
        now = 0
        y = np.array(X).reshape(-1, 5).mean(1)
        for i in y:
            now = max(now, i)
            ys.append(now)
        print(ys)
        return x, ys


    plt.plot(*smooth(Y), label='Our Method', linewidth=5, c='C3')
    plt.plot(*smooth(X), label='Gaussian', linewidth=5, c='C0', linestyle='--')
    plt.legend(loc=2)
    plt.xlim(0, 20)
    plt.ylim(-0.05, 1.05)
    plt.xlabel('Epoch')
    plt.ylabel('Percentage')
    plt.tight_layout()
    plt.title('(D) State Coverage')
    # from tools.utils import plt_save_fig_array
    # img = plt_save_fig_array()
    # images['sac'] = [img] + images['sac']
    # images['ours'] = [img] + images['ours']
    plt.savefig(f'data/maze_occ.png', dpi=300)


    images = dict(
        sac=['mbsac_seed6/50_buffer', 'mbsac_seed6/850_buffer'],
        ours=['gaussian_seed4/50_buffer', 'gaussian_seed4/500_buffer'],
    )
    for i in images:
        #images[i] = [torch.load(f'data/savedbuffer/{i}/buffer.pt') for i in images[i]]
        images[i] = [f'data/maze/{j}.png' for j in images[i]]
    

    images['sac'].append('data/maze/mbsac_seed6/occ_20.png')
    images['ours'].append('data/maze/gaussian_seed4/occ_20.png')

    #from tools.utils import plt_save_fig_array

    group(images)
    plt.tight_layout()
    plt.savefig(f'data/maze.png', dpi=300)


def read_occupancy(path):
    F = f'data/buffers/{path}/traj.mp4'
    import cv2
    import os
    p =f'data/maze/{path}'
    os.makedirs(p, exist_ok=True)
    import pandas as pd
    #csv = open(f'data/buffers/{path}/progress.csv', 'r')
    csv = pd.read_csv(f'data/buffers/{path}/progress.csv')
    
    print(csv['train_occ_metric'].dropna())

    # read video
    cap = cv2.VideoCapture(F)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break

    #return frames, occ
        import cv2
        img = frames[-1][:, 640:640+640][35:-13, 100+4:-100-4]
        print(f'data/maze/{path}/{len(frames)}.png')
        cv2.imwrite(f'data/maze/{path}/occ_{len(frames)}.png', img)


    
if __name__ == '__main__':
    #plot_bandit()


    font = {'size': 40}
    import matplotlib
    import os
    matplotlib.rc('font', **font)

    plot_maze()
    exit(0)
    # mbsac_6
    # for i in range(3, 9):
    #     read_occupancy(f'gaussian_seed{i}')

    read_occupancy(f'gaussian_seed4')
    read_occupancy(f'mbsac_seed6')