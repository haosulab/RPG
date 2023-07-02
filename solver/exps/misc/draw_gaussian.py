import tqdm
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2


def save_fig(std):
    #std = 0.01
    print(std)
    center = np.array([0.2, 0.3, 0.8])
    height = np.array([0.15, 0.15, 0.23])

    def f(x):
        return (np.exp(-(x[:, None] - center)**2/0.04)  + height).max(axis=-1)

    N = 200
    N_S = 100000
    x = np.arange(N)/(N+1)
    ys = []
    for mu in x:
        a = np.random.normal(size=(N_S,)) * std + mu


        y = f(a).mean()
        ys.append(y)

    #fig, ax = plt.subplots(figsize=(4, 4))
    plt.clf()
    ax = plt.gca()

    plt.title(f'E[R] @ std={std:.02f}')

    max_x = x[np.argmax(ys)]

    plt.plot([max_x, max_x], [-2., 2.], color="red", linestyle='dashed')
    plt.plot(x, ys)
    ax.set_ylim([0.2, 1.4])
    ax.set_xlim([0.0, 1.0])
    plt.savefig(f'tmp/tmp.png')
    out = cv2.imread('tmp/tmp.png')[:, :, ::-1]
    return out





if __name__ == '__main__':
    #save_fig(0.3)
    #exit(0)
    #main()
    import imageio
    outs = []
    for std in range(20):
        outs.append(save_fig(std/100.))
        from tools.utils import animate  
        imageio.mimsave('image.gif', outs)