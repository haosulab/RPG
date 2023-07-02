from matplotlib import pyplot as plt

from envs.hiro_robot_envs.maze_env_v2 import MazeEnvV2
from envs.hiro_robot_envs.ant import AntEnv

class NewAntEnv(AntEnv):
    def viewer_setup(self):
        self.viewer.cam.lookat[0] += 3.5
        self.viewer.cam.lookat[1] += 3.5
        self.viewer.cam.lookat[2] = 5
        self.viewer.cam.elevation = -90  # camera rotation around the axis in the plane going through the frame origin (if 0 you just see a line)

class AntMazeEnv(MazeEnvV2):
    MODEL_CLASS = NewAntEnv

env = AntMazeEnv(maze_size_scaling=4, wall_size=0.1)

env.reset()
imgs = []
for i in range(4):
    imgs.append(env.render(mode='rgb_array'))
    env.step(env.action_space.sample())

# name = "wall_0_0_h"
# id = env.wrapped_env.model.geom_name2id(name)
# print(env.wrapped_env.model.geom_pos[id], )

# print(env.wrapped_env.model.geom_pos)
# env.wrapped_env.model.geom_pos[:10] += 10
# env.wrapped_env.sim.forward()

plt.imshow(env.render(mode='rgb_array'))
plt.show()
# env.wrapped_env.model.geom_pos[:10] -= 10
# env.wrapped_env.sim.forward()
