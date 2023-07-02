import numpy as np
from collections import OrderedDict
from .sapien_utils import Pose, sapien, CameraRender, distance, add_link, identity, x2y, load_state_vector, state_vector


SIM = None
OPTIFUSER_RENDERER = None

class SimulatorBase:
    """
    Simulator...
    """

    def __init__(self, dt=0.0025, frameskip=4, gravity=(0, 0, -9.8)):
        self._sim = sapien.Engine()

        self._renderer = sapien.VulkanRenderer(
            default_mipmap_levels=1, device='cuda:0',
        )
        #self._renderer.set_log_level("off")
        self._sim.set_renderer(self._renderer)

        scene_config = sapien.SceneConfig()
        scene_config.sleep_threshold = 0.0000002
        scene_config.gravity = gravity  # set to 0 to make sure the objects stay still when grasping
        scene_config.enable_pcm = False
        scene_config = sapien.SceneConfig()
        scene_config.default_dynamic_friction = 1.0
        scene_config.default_static_friction = 1.0
        scene_config.default_restitution = 0.0
        scene_config.contact_offset = 0.02
        scene_config.enable_pcm = False
        scene_config.solver_iterations = 25
        scene_config.solver_velocity_iterations = 0
        

        self._scene = self._sim.create_scene(config=scene_config)

        self._dt = dt
        self.frameskip = frameskip
        self._viewer = None
        self._viewers = OrderedDict()

        self.metadata = OrderedDict({
            'render.modes': ['human'],
            'video.frames_per_second': int(np.round(1.0 / self._dt))
        })

        self._scene.set_timestep(self._dt)
        self.seed()
        self.objects = []
        self.init_state = None

        self.build_scene()
        self.init_state = self.state_vector()

        self._render_in_simulation = None  # hack to render the intermediate result

    @property
    def dt(self):
        return self._dt * self.frameskip

    def build_scene(self):
        raise NotImplementedError("need to construct scene")

    def state_vector(self):
        return state_vector(self.objects)

    def load_state_vector(self, vec):
        load_state_vector(self.objects, vec)

    def reset(self):
        if self.init_state is not None:
            self.load_state_vector(self.init_state)

    def _get_obs(self):
        raise NotImplementedError

    def seed(self, seed=None):
        from gym.utils import seeding
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_viewer(self, mode):
        self._viewer = self._viewers.get(mode)
        if self._viewer is None:
            if mode == 'human':
                raise NotImplementedError
            elif mode == 'rgb_array':
                self._viewer = CameraRender(self._scene, mode, width=1024, height=780)
            else:
                raise NotImplementedError(f"viewer mode {mode} is not implemented")

            self.viewer_setup()
            if mode == 'human':
                self._viewer.show_window()
            self._viewers[mode] = self._viewer

        self._scene.update_render()
        return self._viewer

    def render(self, mode='human'):
        return self._get_viewer(mode).render()

    def do_simulation(self):
        if self._render_in_simulation is not None:
            self.render(self._render_in_simulation)
        for i in range(self.frameskip):
            self._scene.step()

    def __del__(self):
        self._scene = None

    def viewer_setup(self):
        self._scene.set_ambient_light([.4, .4, .4])
        self._scene.add_point_light([2, 2, 2], [1, 1, 1])
        self._scene.add_point_light([2, -2, 2], [1, 1, 1])
        self._scene.add_point_light([-2, 0, 2], [1, 1, 1])

        self.set_up_camera()
        self._viewer.set_current_scene(self._scene)

    def set_up_camera(self):
        self._viewer.set_camera_position(3, -1.5, 1.65)
        self._viewer.set_camera_rotation(-3.14 - 0.5, -0.2)

    def add_articulation(self, x, y, size, color, ranges, name, friction, damping, material):
        builder = self._scene.create_articulation_builder()
        if isinstance(size, int) or isinstance(size, float):
            size = np.array([size, size, size])

        height = size[-1]
        world = add_link(builder, None, None, name="world")
        yaxis = add_link(builder, world, ([0., 0., 0.], identity), ((0, 0, 0), identity),
                         "x", "xaxis", ranges[0], friction=friction, damping=damping, type='slider')
        xaxis = add_link(builder, yaxis, ([0., 0., 0.], identity), ((0, 0, 0), x2y),
                         "y", "yaxis", ranges[1], friction=friction, damping=damping, type='slider')

        xaxis.add_box_visual(Pose(), size, color, name)
        xaxis.add_box_collision(Pose(), size, material=material, density=1000)
        wrapper = builder.build(True)  # fix base = True
        wrapper.set_root_pose(Pose([0, 0, height + 1e-5]))
        wrapper.set_name(name)
        wrapper.set_qpos([x, y])
        return wrapper

    def add_box(self, x, y, size, color, name, fix=False, shape=True, z=None):
        if isinstance(size, int) or isinstance(size, float):
            size = np.array([size, size, size])
        actor_builder = self._scene.create_actor_builder()
        actor_builder.add_box_visual(Pose(), size, color, name)
        if shape:
            actor_builder.add_box_collision(Pose(), size, density=1000)
        if fix:
            box = actor_builder.build_static(name=name)
        else:
            box = actor_builder.build(name=name)

        if z is None:
            z = size[2] + 1e-5
        pos = Pose(np.array((x, y, z)))
        box.set_pose(pos)
        return box

    def set_up_camera(self):
        self._viewer.set_camera_position(0, 0, 9)
        self._viewer.set_camera_rotation(1.57, -1.57)

    def add_sphere(self, x, y, radius, color, name, fix=False):
        actor_builder = self._scene.create_actor_builder()
        actor_builder.add_sphere_visual(Pose(), radius, color, name)
        actor_builder.add_sphere_collision(Pose(), radius, density=1000)
        if fix:
            box = actor_builder.build_static(name)
        else:
            box = actor_builder.build(name)

        pos = Pose(np.array((x, y, radius + 1e-5)))
        box.set_pose(pos)
        return box