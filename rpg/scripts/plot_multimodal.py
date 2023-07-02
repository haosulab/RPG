import argparse
import os
import numpy as np
import cv2
import numpy as np


def sample_random_pictures(video_path, n):
    # read video to numpy array
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    buf = np.empty((frame_count, frame_height, frame_width, 3), np.dtype('uint8'))
    fc = 0
    ret = True
    path = 'data/tmp'
    os.makedirs(path, exist_ok=True)

    while (fc < frame_count  and ret):
        ret, buf[fc] = cap.read()
        cv2.imwrite(f'data/tmp/{fc}.png', buf[fc])
        fc += 1

    cap.release()

def plot_ant():
    data0 =  "maze_exp/denseantabl/AntPush_denseantabl_seg4save_seed1/2023-01-25-01-12-24-670448/eval*"
    data = 'maze_exp/denseantabl/AntPush_denseantabl_seg4save_seed2/2023-01-25-07-36-56-294096/eval*'

    def get_paths(data):
        #paths = np.array(filter(lambda x: not os.path.isdir(x), glob.glob(data)))
        paths = []
        for i in glob.glob(data):
            if os.path.isdir(i):
                paths.append(i)
        paths = np.array(paths)
        idx = np.array([int(i.split('eval')[-1]) for i in paths]).argsort()
        paths = paths[idx]
        return paths

    import glob
    import numpy as np

    
    font = {'size': 20}
    import matplotlib
    import os
    matplotlib.rc('font', **font)

    #paths = paths[[0,4,8,16]]
    paths = get_paths(data0)[[0, 4, 8, 16]]
    paths[3] = get_paths(data)[16]

    images = []

    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 6))
    import cv2

    img = cv2.imread('data/envs/AntPush.png')[..., ::-1]
    print(img.shape)
    plt.imshow(img[0:300, 100:400, [0,1,2]])
    plt.title('AntPush Environment')
    plt.axis('off')

    from tools.utils import plt_save_fig_array
    img = plt_save_fig_array()
    images.append(img)


    for i in paths:
        import torch
        print(i)
        traj = torch.load(os.path.join(i, 'traj.pt'))
        #traj = agent.evaluate(None, 200)
        next_obs = traj.get_tensor('next_obs')
        z = traj.get_tensor('z')

        import matplotlib.pyplot as plt
        plt.clf()
        plt.figure(figsize=(6, 6))
        oo = next_obs[..., :2].reshape(-1, 2).detach().cpu().numpy()
        z = z.reshape(-1).detach().cpu().numpy()
        C = np.array(['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C1', 'C2', 'C3'])
        plt.scatter(oo[:, 0], oo[:, 1], s=2, c='#FFE4B5')
        plt.xlim(-10., 10.)
        plt.ylim(-4., 16.)
        plt.title(f"Episode {int(i.split('eval')[-1])}")

        img = plt_save_fig_array()
        images.append(img)
    out = np.concatenate(images, axis=1)
    import cv2
    print(out.shape)
    cv2.imwrite('data/ant_explore.png', out[..., [2,1,0]])

    
def plot_others(env_name):
    import data
    import glob
    data =  sorted(glob.glob(f"data/savedeval/{env_name}/*.png"))
    print(data)

    font = {'size': 40}
    import matplotlib
    import os
    matplotlib.rc('font', **font)


    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 6))

    img = cv2.imread(data[0])[..., ::-1]
    plt.imshow(img)
    X = env_name.replace('block', 'BlockPush').replace('Cabinet', 'Cabinet (sparse)').replace('stickpull', 'StickPull')
    plt.title(f'{X}')
    plt.axis('off')


    from tools.utils import plt_save_fig_array
    images = []
    img = plt_save_fig_array()
    images.append(img)

    for p in data[1:]:
        img = cv2.imread(p)[..., ::-1]
        plt.imshow(img)
        plt.title('')
        plt.axis('off')
        img = plt_save_fig_array()
        images.append(img)

    cv2.imwrite(f'data/{env_name}_explore.png', np.concatenate(images, axis=1)[..., [2,1,0]])
    

if __name__ == '__main__':
    #plot_ant()
    #exit(0)
    plot_others('stickpull')
    plot_others('block')
    plot_others('cabinet')
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--video_path', type=str, default=None)
    # parser.add_argument('--n', type=int, default=5)
    # args = parser.parse_args()


    # img0, imgs = sample_random_pictures(args.video_path, args.n)
    # img = np.concatenate([img0] + list(imgs), axis=1)

    # cv2.imwrite('data/random_pictures.png', img)