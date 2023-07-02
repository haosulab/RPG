# usage: load your urdf and edit it on the fly
# if your urdf is buggy this script will print out error message
# otherwise this script detects urdf file change and reloads your robot

from solver.envs.rigidbody3d.rigid3d_simulator import Rigid3dSim, SapienViewer

class TestURDF(Rigid3dSim):

    def __init__(self, urdf_path, dt=0.001, frame_skip=0) -> None:
        super().__init__(dt, frame_skip)
        self.arm = self.load_urdf(urdf_path)


if __name__ == "__main__":
    import cv2
    import hashlib
    import time

    urdf_path = "gripper_v2.urdf"

    current = None
    current_articulation = None
    viewer = SapienViewer(resolution=(128, 128), distance=3)
    viewer.create_window()

    while True:
        with open(urdf_path, "rb") as f:
            h = hashlib.md5(f.read()).hexdigest()
            if h != current:
                current = h
                print("================================================================================================================")
                print(current)
                try:
                    env = TestURDF(urdf_path)
                    cv2.imwrite("tests/test_urdf/debug.png", env.render())
                except Exception as e:
                    print(e)
                if current_articulation is not None:
                    viewer.scene.remove_articulation(current_articulation)
                current_articulation = viewer.load_urdf(urdf_path)
                viewer.load_urdf("./assets/ground.urdf")
            viewer.window.render()

