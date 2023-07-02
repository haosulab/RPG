import sys
sys.path.append(".")

from rigid3d_simulator import Rigid3dSim
import torch


class TestCollideEnv(Rigid3dSim):

    def __init__(self, dt=0.001, frame_skip=0) -> None:
        super().__init__(dt, frame_skip)
        self.world.setGravity([0, 0, 0])
        self.ball = self.load_urdf("./assets/sphere.urdf")
        self.ground = self.load_urdf("./assets/ground.urdf")
        self.ball.nimble_actor.getBodyNode(0).setRestitutionCoeff(1)
        self.ground.nimble_actor.getBodyNode(0).setRestitutionCoeff(1)

    def get_init_state(self):
        return torch.tensor(
            [0, 0, 0, 0, -0.3, 0, 0, 0, 0, 0, -1, 0], requires_grad=True
        )


if __name__ == "__main__":
    from tools.utils import animate
    import tqdm

    env = TestCollideEnv()
    state = env.reset()
    initial_state = state

    imgs = []
    for i in range(200):
        img = env.render(text=f"h={state[4].item():.3f}")
        imgs.append(img)
        print(state[:6].tolist())
        state, reward, done, info = env.step(torch.zeros(env.world.getActionSize()))
    
    print("check gradient")
    state[4].backward()
    print("dhT/dh0", initial_state.grad[4])

    animate(imgs, "tests/test_gradient/video.mp4", fps=20)
