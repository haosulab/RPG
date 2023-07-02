import argparse
import numpy as np
from tools.utils import read_video



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('--out', default='out.png')
    args = parser.parse_args()

    frames = read_video(args.path)
    
    import cv2
    #cv2.imwrite('init.png', frames[0][:, :, ::-1])

    mid = frames[45]
    final = frames[-1]
    x = final

    cv2.imwrite('mid.png', (mid[:, :, [2, 1, 0]] * 255).astype(np.uint8))
    cv2.imwrite(args.out, (x[:, :, [2, 1, 0]] * 255).astype(np.uint8))
    # x = frames[0]
    # num = 1
    # for i in range(1, len(frames), 10):
    #     x = x + frames[i]
    #     num = num + 1

    # x = x/num

    # x = x * 0.7 + frames[-1] * 0.3
    # # x = np.maximum(x, frames[-1])
    # cv2.imwrite(args.out, (x[:, :, ::-1] * 255).astype(np.uint8))