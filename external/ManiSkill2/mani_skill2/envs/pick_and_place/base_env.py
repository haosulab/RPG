from typing import Type, Union

import numpy as np
import sapien.core as sapien
from sapien.core import Pose

from mani_skill2.agents.robots.panda import Panda
from mani_skill2.agents.robots.xmate3 import Xmate3Robotiq
from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.utils.sapien_utils import (
    get_entity_by_name,
    look_at,
    set_articulation_render_material,
    vectorize_pose,
)


class StationaryManipulationEnv(BaseEnv):
    SUPPORTED_ROBOTS = {"panda": Panda, "xmate3_robotiq": Xmate3Robotiq}
    agent: Union[Panda, Xmate3Robotiq]

    def __init__(self, *args, robot="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_uuid = robot
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, **kwargs)

    def _build_cube(
        self,
        half_size,
        color=(1, 0, 0),
        name="cube",
        static=False,
        render_material: sapien.RenderMaterial = None,
    ):
        if render_material is None:
            render_material = self._renderer.create_material()
            render_material.set_base_color(np.hstack([color, 1.0]))

        builder = self._scene.create_actor_builder()
        builder.add_box_collision(half_size=half_size)
        builder.add_box_visual(half_size=half_size, material=render_material)
        if static:
            return builder.build_static(name)
        else:
            return builder.build(name)

    def _build_sphere_site(self, radius, color=(0, 1, 0), name="goal_site"):
        """Build a sphere site (visual only). Used to indicate goal position."""
        builder = self._scene.create_actor_builder()
        builder.add_sphere_visual(radius=radius, color=color)
        sphere = builder.build_static(name)
        # NOTE(jigu): Must hide after creation to avoid pollute observations!
        sphere.hide_visual()
        return sphere

    def _load_agent(self):
        agent_cls: Type[Panda] = self.SUPPORTED_ROBOTS[self.robot_uuid]
        self.agent = agent_cls(self._scene, self._control_freq, self._control_mode)
        self.tcp: sapien.Link = get_entity_by_name(
            self.agent.robot.get_links(), self.agent._config.ee_link_name
        )
        set_articulation_render_material(self.agent.robot, specular=0.9, roughness=0.3)

    def _initialize_agent(self):
        if self.robot_uuid == "panda":
            # fmt: off
            # EE at [0.615, 0, 0.17]
            qpos = np.array(
                [0.0, np.pi / 8, 0, -np.pi * 5 / 8, 0, np.pi * 3 / 4, np.pi / 4, 0.04, 0.04]
            )
            # fmt: on
            qpos[:-2] += self._episode_rng.normal(
                0, self.robot_init_qpos_noise, len(qpos) - 2
            )
            self.agent.reset(qpos)
            self.agent.robot.set_pose(Pose([-0.615, 0, 0]))
        elif self.robot_uuid == "xmate3_robotiq":
            qpos = np.array(
                [0, np.pi / 6, 0, np.pi / 3, 0, np.pi / 2, -np.pi / 2, 0, 0]
            )
            qpos[:-2] += self._episode_rng.normal(
                0, self.robot_init_qpos_noise, len(qpos) - 2
            )
            self.agent.reset(qpos)
            self.agent.robot.set_pose(Pose([-0.562, 0, 0]))
        else:
            raise NotImplementedError(self.robot_uuid)

    def _initialize_agent_v1(self):
        """Higher EE pos."""
        if self.robot_uuid == "panda":
            # fmt: off
            qpos = np.array(
                [0.0, 0, 0, -np.pi * 2 / 3, 0, np.pi * 2 / 3, np.pi / 4, 0.04, 0.04]
            )
            # fmt: on
            qpos[:-2] += self._episode_rng.normal(
                0, self.robot_init_qpos_noise, len(qpos) - 2
            )
            self.agent.reset(qpos)
            self.agent.robot.set_pose(Pose([-0.615, 0, 0]))
        elif self.robot_uuid == "xmate3_robotiq":
            qpos = np.array([0, 0.6, 0, 1.3, 0, 1.3, -1.57, 0, 0])
            qpos[:-2] += self._episode_rng.normal(
                0, self.robot_init_qpos_noise, len(qpos) - 2
            )
            self.agent.reset(qpos)
            self.agent.robot.set_pose(Pose([-0.562, 0, 0]))
        else:
            raise NotImplementedError(self.robot_uuid)

    def _setup_cameras(self):
        # Camera only for rendering, not included in `_cameras`
        self.render_camera = self._scene.add_camera(
            "render_camera", 512, 512, 1.5, 0.01, 10
        )
        self.render_camera.set_local_pose(look_at([1.0, 1.0, 0.8], [0.0, 0.0, 0.5]))

        base_camera = self._scene.add_camera(
            "base_camera", 128, 128, np.pi / 2, 0.01, 10
        )
        base_camera.set_local_pose(look_at([0.2, 0, 0.4], [0, 0, 0]))
        self._cameras["base_camera"] = base_camera

    def _setup_viewer(self):
        super()._setup_viewer()
        self._viewer.set_camera_xyz(0.8, 0, 1.0)
        self._viewer.set_camera_rpy(0, -0.5, 3.14)

    def _get_obs_agent(self):
        obs = self.agent.get_proprioception()
        obs["base_pose"] = vectorize_pose(self.agent.robot.pose)
        return obs
