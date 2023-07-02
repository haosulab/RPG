import time
import tqdm
from r3d_base import Rigid3dBase

env = Rigid3dBase(cam_resolution=(256, 256))
env.reset()
import cv2
cv2.imwrite("debug.png", env.render())