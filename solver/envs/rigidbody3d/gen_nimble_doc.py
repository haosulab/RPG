from asyncore import write
import code
import os
import torch
from tools.utils import animate
import tqdm
import time
from r3d_base import Rigid3dBase
from r3d_base import Rigid3dBase
import nimblephysics as nimble
from utils import arr_to_str


class TestIkEnv(Rigid3dBase):

    def __init__(self, cfg=None, 
        # low_steps=200, dt=0.005, frame_skip=0, 
        # friction_coeff=0.3, n_batches=1, 
        # X_OBS_MUL=5., V_OBS_MUL=5., A_ACT_MUL=1.
    ):
        super().__init__(cfg)
        # super().__init__(cfg, low_steps, dt, frame_skip, friction_coeff, n_batches, X_OBS_MUL, V_OBS_MUL, A_ACT_MUL)

    def init_simulator(self):
        self.sim.arm    = self.sim.load_urdf("beginner.urdf"  , friction_coeff=self.friction_coeff)
        self.sim.ball   = self.sim.load_urdf("box.urdf"    , friction_coeff=self.friction_coeff)
        self.sim.ground = self.sim.load_urdf("ground.urdf" , friction_coeff=self.friction_coeff)
        # action only control arm, not other objects
        for i in range(self.sim.arm.sapien_actor.dof, self.sim.world.getActionSize()):
            self.sim.world.removeDofFromActionSpace(i)
        # nimble forward kinematics
        self.sim.arm_fk = nimble.neural.IKMapping(self.sim.world)
        self.sim.arm_fk.addLinearBodyNode(self.sim.arm.nimble_actor.getBodyNode("end_effector"))

    def sample_state_goal(self, batch_size=1):
        state = torch.tensor(
            [   
                # arm qpos
                -0.2, 0.4, 0, 0.2,
                # ball/box pos: [expcoord, xyz]
                0, 1.57 * 0.5, -1.57 * 0.7, 0, 0, 1,
                # velocities
                0, 0, 0, 0,
                0, 0, 0, 0, 0, 0
            ], dtype=torch.float32
        )
        goals = torch.tensor([0, 0, -3], dtype=torch.float32)
        return state, goals


if __name__ == "__main__":
    from utils import output_help_to_file

    env = TestIkEnv()
    env.reset()

    outdir = "nimble_help_docs"
    os.makedirs(outdir, exist_ok=True)

    def write_help(obj):
        output_help_to_file(f"{outdir}/{str(type(obj).__name__)}.txt", obj)
        print(str(type(obj).__name__))

    os.system("rm nimble_help_docs/*")
    write_help(nimble)
    write_help(env.sim.world)
    write_help(env.sim.arm.nimble_actor)
    # write_help(env.sim.arm.nimble_actor.getJoints())
    node = env.sim.arm.nimble_actor.getBodyNode("end_effector")
    write_help(node)