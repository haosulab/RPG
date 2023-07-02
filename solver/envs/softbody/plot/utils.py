# merge images
from tools.utils import animate 
import numpy as np

def fuse(images, t, method):
    if method == 'max':
        s = None
        for i in images:
            if s is None:
                s = i[t]
            else: 
                s = np.maximum(s, i[t])
    elif method == 'mean':
        s = 0
        for i in images:
            s = s + i[t].astype(np.float32)
        s = s / len(images)
    elif method == 'concat':
        s = np.concatenate([i[t] for i in images], 1)

    return s.clip(0, 255).astype(np.uint8)

def fuse_traj_images(images, target=None, method='mean'):
    im = []
    for t in range(len(images[0])):
        im.append(fuse(images, t, method))
    if target is not None:
        animate(im, target)
    return im