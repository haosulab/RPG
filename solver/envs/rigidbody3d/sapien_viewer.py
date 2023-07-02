import numpy as np
import sapien.core as sapien
import transforms3d.euler
from sapien.utils import Viewer


class SapienViewer:
    
    def __init__(self, resolution, distance) -> None:
        self.engine = sapien.Engine()
        self.renderer = sapien.VulkanRenderer()
        # self.renderer = sapien.KuafuRenderer(sapien.KuafuConfig())
        self.engine.set_renderer(self.renderer)
        
        self.scene = self.engine.create_scene()
        self.render_scene = self.scene.get_renderer_scene()
        self.setup_lighting()

        self.camera = self.setup_camera(resolution, distance)
        self.window = None

    def setup_lighting(self):        
        # self.scene.set_ambient_light([0.5, 0.5, 0.5])
        # self.scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])
        # it's easier to visualize rotation with more directional light
        self.scene.set_ambient_light([0.1, 0.1, 0.1])
        self.scene.add_directional_light([0, 1, -1], [2, 2, 2])

    def setup_camera(self, resolution, distance):
        near, far = 0.05, 100
        width, height = resolution
        camera_mount_actor = self.scene.create_actor_builder().build_kinematic()
        self.camera_mount_actor = camera_mount_actor
        camera = self.scene.add_mounted_camera(
            name="camera",
            actor=camera_mount_actor,
            pose=sapien.Pose(),  # relative to the mounted actor
            width=width,
            height=height,
            fovx=np.deg2rad(35),
            fovy=np.deg2rad(35),
            near=near,
            far=far,
        )
        camera_mount_actor.set_pose(
            sapien.Pose(
                [0, -2 * distance, distance],
                transforms3d.euler.euler2quat(0, np.arctan2(2, 4), np.pi / 2)
            )
        )
        return camera

    def take_picture(self):
        self.camera.take_picture()
        rgba = self.camera.get_float_texture('Color')
        return (rgba * 255).clip(0, 255).astype("uint8")

    def create_window(self):
        self.window = Viewer(self.renderer)
        # pylance thinks this line is an infinite loop, very weird
        eval("self.window.set_scene(self.scene)")
        self.window.set_camera_xyz(x=-4, y=0, z=2)
        self.window.set_camera_rpy(r=0, p=-np.arctan2(2, 4), y=0)
        self.window.window.set_camera_parameters(near=0.05, far=100, fovy=1)

    def close_window(self):
        self.window.close()
        self.window = None

    def load_urdf(self, urdf_path, root_pose=None, fix_root_link=False):
        sapien_loader = self.scene.create_urdf_loader()
        sapien_loader.fix_root_link = fix_root_link
        sapien_actor = sapien_loader.load(urdf_path)
        if root_pose is None:
            root_pose = sapien.Pose(
                [0, 0, 0], 
                transforms3d.euler.euler2quat(
                    np.pi / 2, 0, -np.pi / 2
                )
            )
        sapien_actor.set_root_pose(root_pose)
        return sapien_actor
