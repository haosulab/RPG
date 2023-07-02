import os
import numpy as np
from multiprocessing import Process


env_name = dict(
    cabinet='EEArm',
    stickpull='MWStickPull',
    kitchen='Kitchen',
    hammer='AdroitHammer',
    antpush='AntPush',
    ant2='AntPush2',
    block='BlockPush2',
    fall='AntFall',
    block3='BlockPush',
    door='AdroitDoor',
    basket='MWBasketBall',
)



def render_worker(name):
    filename = 'data/envs/{}.png'.format(name)
    if os.path.exists(filename):
        return

    from rpg.env_base import make
    import cv2
    kwargs = {} 

    if name.startswith('Adroit'):
        kwargs['img_size'] = 512
    env = make(name, n=1, **kwargs)
    env.start()
    img = env.render('rgb_array')
    cv2.imwrite(filename, img[0][..., ::-1])



if __name__ == '__main__':
    os.makedirs('data/envs', exist_ok=True)
    envs = ['hammer', 'door', 'basket', 'stickpull', 'block', 'cabinet', 'kitchen', 'antpush'] #, 'antfall']
    for name in envs:
        p = Process(target=render_worker, args=(env_name[name],))
        p.start()
        p.join() 

    from matplotlib import pyplot as plt
    font = {'size': 24}
    import matplotlib
    import os
    matplotlib.rc('font', **font)

    images = [plt.imread('data/envs/{}.png'.format(env_name[i]))[..., :3] for i in envs]

    width = min(10, len(envs))
    n_rows = (len(envs) + width - 1)//width

    fig, axs = plt.subplots(n_rows, width, figsize=(6 * width, 6 * n_rows))
    if isinstance(axs[0], np.ndarray):
        axs = sum([list(x) for x in axs], [])

    idx = ord('A')
    for ax, name, img in zip(axs, envs, images):
        #print(img.shape)
        if img.shape[0] != img.shape[1]: 
            w = min(img.shape[:2])
            img = img[img.shape[0]//2-w//2:img.shape[0]//2+w//2, img.shape[1]//2-w//2:img.shape[1]//2+w//2]
        ax.imshow(img)
        ax.set_title(f"({chr(idx)}) {name}")
        ax.axis('off')
        idx += 1
    plt.tight_layout()
    plt.savefig('data/sparse_envs.png')