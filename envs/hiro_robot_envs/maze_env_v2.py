# build a maze with less thiner walls... the current one is not very efficient..

import os
import numpy as np

"""Adapted from hiro-robot-envs maze_env.py."""

import os
import tempfile
import xml.etree.ElementTree as ET
import numpy as np
import gym

from . import maze_env_utils

# Directory that contains mujoco xml files.
MODEL_DIR = 'assets'


class MazeEnvV2(gym.Env):
    MODEL_CLASS = None

    MAZE_HEIGHT = None
    MAZE_SIZE_SCALING = None

    def __init__(
            self,
            width=4,
            height=4,
            maze_id=None,
            maze_height=0.5,
            wall_size = 0.05,
            maze_size_scaling=8,
            maze_type=None,
            *args, **kwargs
    ):
        self._maze_id = maze_id
        self.wall_size = wall_size
        self.t = 0

        self.MAZE_HEIGHT = maze_height
        self.MAZE_SIZE_SCALING = maze_size_scaling

        self.width = width
        self.height = height


        model_cls = self.__class__.MODEL_CLASS
        if model_cls is None:
            raise Exception("MODEL_CLASS unspecified!")
        xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_DIR, model_cls.FILE)
        tree = ET.parse(xml_path)

        worldbody = tree.find(".//worldbody")

        self.maze_type = maze_type

        if maze_type is None or maze_type == 'regular':
            for i in range(width+1):
                for j in range(height+1):
                    if i != width:
                        self.add_wall(worldbody, f"wall_{i}_{j}_h", i, j, 0)
                    if j != height:
                        self.add_wall(worldbody, f"wall_{i}_{j}_v", i, j, 1)
        elif maze_type.startswith('cross'):
            w = self.MAZE_SIZE_SCALING
            h = self.MAZE_SIZE_SCALING * 2
            x = h + w/2

            if maze_type == 'cross':
                pts = [[-x, x], [-w/2, x] , [-w/2, x + h], [-x, x + h], [-x, x+h+w], [-w/2, x+h+w], [-w/2, x+ 2*h + w]]
            else:
                pts = [[-w/2, x] , [-w/2, x + h], [-x, x + h], [-x, x+h+h*2 + w], [-w/2, x+h+h*2 + w]]
                pts = [[i[0], i[1] -  2*w]for i in pts]

            pts += [[-i[0], i[1]]for i in pts[::-1]]
            pts += [[i[1], i[0]]for i in pts[-2::-1]]
            pts += [[-i[1], -i[0]]for i in pts[-2:0:-1]]

            if maze_type == 'cross':
                anchors = [[0, 2*x], [-w, 2*x], [-h, 2*x], [w, 2*x], [h, 2*x], [0, 2 * x + w], [0, 2*x + h], [0, 2 * x - w], [0, 2*x - h]]
            else:
                anchors = [[0, w * 1 ], [0, w * 2]]
                for i in [-h, -w, 0, w, h]:
                   for j in [-h, -w, 0, w, h]:
                       anchors.append([i, j + w * 5])
                anchors.append([0., 0.])

            anchors += [[i[1], i[0]]for i in anchors]
            anchors += [[-i[1], -i[0]]for i in anchors]
            
            for i in [-h, -w, 0, w, h]:
                for j in [-h, -w, 0, w, h]:
                    anchors.append([i, j])

            self.anchors = np.array(anchors)
            # values = np.random.random(len(anchors))

            # xy = np.int64(self.anchors / self.MAZE_SIZE_SCALING)
            # from solver.draw_utils import plot_grid_point_values
            # import matplotlib.pyplot as plt
            # plot_grid_point_values(xy, values)
            # plt.savefig("test.png")
            # exit(0)
            
            # print('saving test.png ')
            # aa = np.array(anchors) / self.MAZE_SIZE_SCALING
            # import matplotlib.pyplot as plt
            # pp = np.array(pts) / self.MAZE_SIZE_SCALING
            # fig, ax = plt.subplots(figsize=(5, 5))
            # plt.plot([i[0] for i in pp], [i[1] for i in pp])
            # plt.scatter([i[0] for i in aa], [i[1] for i in aa])
            # plt.savefig("test.png")
            # exit(0)

            for i in range(len(pts)):
                j = (i + 1) % len(pts)
                self.add_obstacles(worldbody, f"wall_{i}", np.array(pts[i]), np.array(pts[j]))
        else:
            raise NotImplementedError

        torso = tree.find(".//body[@name='torso']")
        geoms = torso.findall(".//geom")
        for geom in geoms:
            if 'name' not in geom.attrib:
                raise Exception("Every geom of the torso must have a name "
                                "defined")

        _, file_path = tempfile.mkstemp(text=True, suffix=".xml")
        tree.write(file_path)

        self.wrapped_env = model_cls(*args, file_path=file_path, **kwargs)

        self._init_poses = self.wrapped_env.model.geom_pos[:].copy()

    def add_obstacles(self, worldbody, name, p1, p2):
        scale = self.MAZE_SIZE_SCALING

        mid = (p1 + p2) / 2
        pos = "%f %f %f" % (mid[0], mid[1], self.MAZE_HEIGHT / 2 * scale)

        width = height = self.wall_size * scale
        if np.allclose(p1[0], p2[0]):
            assert p2[1] != p1[1]
            height += abs(p2[1] - p1[1]) / 2
        elif np.allclose(p1[1], p2[1]):
            assert p2[0] != p1[0]
            width += abs(p2[0] - p1[0]) / 2
        else:
            raise NotImplementedError

        size = "%f %f %f" % (width, height, self.MAZE_HEIGHT / 2 * scale)

        ET.SubElement(
            worldbody, "geom", name=name, pos=pos, size=size,
            type="box", material="", contype="1", conaffinity="1", rgba="0.4 0.4 0.4 1",
        )

    def add_wall(self, worldbody, name, x, y, type=0):
        scale = self.MAZE_SIZE_SCALING

        if type == 0:
            pos = "%f %f %f" % ((x + 0.5) * scale, y * scale, self.MAZE_HEIGHT / 2 * scale)
            size = "%f %f %f" % ((0.5 + self.wall_size) * scale, self.wall_size * scale, self.MAZE_HEIGHT / 2 * scale)
        else:
            pos = "%f %f %f" % (x * scale, (y + 0.5) * scale, self.MAZE_HEIGHT / 2 * scale)
            size = "%f %f %f" % (self.wall_size * scale, (0.5 + self.wall_size) * scale, self.MAZE_HEIGHT / 2 * scale)

        ET.SubElement(
            worldbody, "geom", name=name, pos=pos, size=size,
            type="box", material="", contype="1", conaffinity="1", rgba="0.4 0.4 0.4 1",
        )

    def set_map(self, map):
        if self.maze_type == 'regular' or self.maze_type is None:
            poses = self._init_poses.copy()

            for i in range(self.width):
                for j in range(self.height):
                    if not map[0, j, i]:
                        name = f"wall_{i}_{j}_h"
                        id = self.wrapped_env.model.geom_name2id(name)
                        poses[id][:2] -= 200
                    if not map[2, j, i]:
                        name = f"wall_{i}_{j}_v"
                        id = self.wrapped_env.model.geom_name2id(name)
                        poses[id][:2] -= 200
            self.wrapped_env.model.geom_pos[:] = poses
            self.wrapped_env.sim.forward()

    def _get_obs(self):
        return np.concatenate([self.wrapped_env._get_obs(),
                               [self.t * 0.001]])

    def reset(self):
        self.t = 0
        self.wrapped_env.reset()
        return self._get_obs()

    @property
    def viewer(self):
        return self.wrapped_env.viewer

    def render(self, *args, **kwargs):
        return self.wrapped_env.render(*args, **kwargs)

    @property
    def observation_space(self):
        shape = self._get_obs().shape
        high = np.inf * np.ones(shape)
        low = -high
        return gym.spaces.Box(low, high)

    @property
    def action_space(self):
        return self.wrapped_env.action_space

    def step(self, action):
        self.t += 1
        inner_next_obs, inner_reward, done, info = self.wrapped_env.step(action)
        next_obs = self._get_obs()
        done = False
        return next_obs, inner_reward, done, info
