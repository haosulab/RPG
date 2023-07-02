import cv2
import torch
import nimblephysics as nimble
from tools.utils import animate
from utils import arr_to_str
from r3d_base import Rigid3dBase

class TestRoll(Rigid3dBase):

    def __init__(self, cfg=None, 
        low_steps=200, dt=0.005, frame_skip=0, 
        friction_coeff=0.3, n_batches=1, 
        X_OBS_MUL=5., V_OBS_MUL=5., A_ACT_MUL=1.
    ):
        super().__init__(cfg, low_steps, dt, frame_skip, friction_coeff, n_batches, X_OBS_MUL, V_OBS_MUL, A_ACT_MUL)

    def init_simulator(self):
        self.sim.ball   = self.sim.load_urdf("./assets/box.urdf"    , friction_coeff=self.friction_coeff)
        self.sim.ground = self.sim.load_urdf("./assets/ground.urdf" , friction_coeff=self.friction_coeff)
    
    def get_reward(self, s, a, s_next):
        return 0

    def sample_state_goal(self, batch_size=1):
        # give box a initial rotation
        state = torch.tensor(
            [
                0, 1.57 * 0.5, -1.57 * 0.7, 0, 0, 1,
                0, 0, 0, 0, 0, 0
            ], 
            dtype=torch.float32
        )
        return state, torch.zeros([0, 0, 0])


if __name__ == "__main__":
    path = "tests/test_box_roll"
    env = TestRoll()
    s = env.reset()

    cv2.imwrite(f"{path}/debug.png", env.render())
    
    imgs = []
    for _ in range(200):
        print(arr_to_str(s))
        imgs.append(env.render())
        s, _, _, _ = env.step(torch.zeros(env.action_space.shape[0]))
    animate(imgs, f"{path}/video.mp4", fps=20)

