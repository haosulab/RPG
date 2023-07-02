import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt


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


def myplot(traj, method):
    traj = traj[0]
    states = torch.cat([traj['s'], traj['last_obs'][None,:]], axis=0)
    # states = states.reshape(-1, 3)
    from solver.draw_utils import plot_colored_embedding
    img = plot_background()
    # plt.imshow(img)
    # plt.savefig('xx.png')
    # exit(0)

    plt.clf()
    plt.figure(figsize=(10, 10))
    if traj['z'].max() < 100 or traj['z'].dtype != torch.int64:
        plt.imshow(np.uint8(img[...,::-1]))
        plot_colored_embedding(traj['z'], states[1:, :, :2] * 512 + 256, s=30)
    else:
        plt.imshow(img[...,::-1])
        states = states.reshape(-1, 3)
        plt.scatter(states[:, 0], states[:, 1], s=30)
    plt.xlim([256, 512+256])
    plt.ylim([256, 512+256])
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(method + '_states.png')

    plt.clf()
    plt.figure(figsize=(8, 6))
    plot_colored_embedding(traj['z'], traj['a'])
    plt.axis('on')
    plt.tight_layout()
    plt.savefig(method + '_actions.png')



traj = torch.load('../../numerical/tmp/latest_traj.th')
myplot(traj, 'seq')

traj = torch.load('../../numerical/tmp2/latest_traj.th')
myplot(traj, 'traj')

traj = torch.load('../../numerical/tmp3/latest_traj.th')
myplot(traj, 'gaussian')