import cv2
import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 6))

from tools.utils import plt_save_fig_array

def plot_background():
    size = 512
    start = size//2

    #x, y = self.state[index][:2].detach().cpu().numpy()
    img = np.zeros((size*2, size*2, 3), dtype=np.uint8)

    img = cv2.circle(img, (int(0.25*size)+start, int(0.25*size)+start), int(0.15 * size), (0, 255, 0), -1)
    img = cv2.circle(img, (int(0.25*size)+start, int(0.75*size)+start), int(0.15 * size), (0, 255, 0), -1)
    img = cv2.circle(img, (int(0.75*size)+start, int(0.75*size)+start), int(0.15 * size), (0, 255, 0), -1)
    img = cv2.circle(img, (int(0.75*size)+start, int(0.25*size)+start), int(0.15 * size), (0, 255, 0), -1)

    img = cv2.rectangle(img, (start, start), (size+start, size+start), (255, 0, 255), 1, 1)
    #img = cv2.circle(img, (int(x*size)+start, int(y*size)+start), 3, (255, 0, 0), -1)

    # img = np.zeros((size*2, size*2, 3), dtype=np.uint8)
    # img = cv2.circle(img, (int(x*size)+start, int(y*size)+start), 3, (255, 0, 0), -1)
    # img = cv2.circle(img, (int(0.5*size)+start, int(0.5*size)+start), int(0.4 * size), (0, 255, 0), -1)
    # img = cv2.rectangle(img, (start, start), (size+start, size+start), (255, 0, 255), 1, 1)
    return img



def myplot(states, method):
    from solver.draw_utils import plot_colored_embedding
    img = plot_background()
    states = states * 512 + 256

    plt.clf()
    plt.imshow(img[...,::-1])
    plt.scatter(states[:, 0], states[:, 1], s=30, c='r')
    plt.xlim([256, 512+256])
    plt.ylim([256, 512+256])
    plt.axis('off')
    plt.tight_layout()
    state = plt_save_fig_array()/255.
    return state


def plot_rl():
    import cv2
    traj = torch.load(f'../../numerical/single/traj1.th')[0]

    states = torch.cat([traj['s'], traj['last_obs'][None,:]], axis=0)
    images = []
    import tqdm
    all = None
    for s in tqdm.tqdm(states, total=len(states)):
        img = myplot(s.detach().cpu().numpy(), 'rrt')

        if all is None:
            images.append(img)
            all = img
        else:
            images.append(np.maximum(img, all * 0.3))
            all = np.maximum(all, img)
        # img = myplot(s.detach().cpu().numpy(), None)
    #print(img.shape)
    #cv2.imwrite('single.png', img[..., [2, 1, 0]])
    from tools.utils import animate
    animate(images, 'single.gif', fps=10)

def plot_rrt():
    from rrt import get_tree
    path, trees = get_tree()
    path = np.array(path) * 512 + 256

    img = plot_background()

    trees = np.array(trees) * 512 + 256

    plt.clf()
    #for a, b in zip(trees[:, 0], trees[:, 1]):
    plt.axis('off')
    plt.imshow(img[...,::-1])
    plt.xlim([256, 512+256])
    plt.ylim([256, 512+256])

    imgs = []
    for idx, line in enumerate(trees):
        plt.plot(line[:, 0], line[:, 1], 'b', linewidth=2)
        if idx % 10 == 0:
            plt.tight_layout()
            imgs.append(plt_save_fig_array()/255.)
    #plt.scatter(trees[:, 0, 0], trees[:, 0, 1], c='b', s=20)

    plt.plot(path[:, 0], path[:, 1], 'r', linewidth=4)

    plt.scatter(path[:, 0], path[:, 1], s=30, c='r')
    plt.tight_layout()
    found_path = plt_save_fig_array()/255.

    # import cv2
    # cv2.imwrite('rrt.png', found_path[..., [2, 1, 0]]*255)
    imgs += [found_path] * len(imgs)

    from tools.utils import animate
    animate(imgs, 'rrt.gif', fps=10)




# plot_rrt()

img = plot_background()

plt.axis('off')
plt.imshow(img[...,::-1])
plt.xlim([256, 512+256])
plt.ylim([256, 512+256])
plt.tight_layout()
plt.savefig('background.png')