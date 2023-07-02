import numpy as np
import sapien.core as sapien
from sapien.core import Pose
import transforms3d
import numpy as np

from transforms3d.quaternions import qmult, rotate_vector, axangle2quat, mat2quat

identity = np.array([1, 0, 0, 0])
x2y = np.array([0.7071068, 0, 0, 0.7071068])
x2z = np.array([0.7071068, 0, 0.7071068, 0])


def add_link(builder, father, link_pose, local_pose=None, name=None, joint_name=None, range=None,
             friction=0., damping=0., type='hinge', father_pose_type='mujoco', contype=1, conaffinity=1):
    # range  [a, b]
    link: sapien.LinkBuilder = builder.create_link_builder(father)
    link.set_name(name)

    if father is not None:
        assert type in ['hinge', 'slider']
        link_pose = np.array(link_pose[0]), np.array(link_pose[1])
        local_pose = np.array(local_pose[0]), np.array(local_pose[1])

        def parent_pose(xpos, xquat, ypos, yquat):
            pos = rotate_vector(ypos, xquat) + xpos
            quat = qmult(xquat, yquat)
            return Pose(pos, quat)

        if type == 'hinge':
            joint_type = 'revolute'
        else:
            joint_type = 'prismatic'

        link.set_joint_name(joint_name)
        father_pose = parent_pose(*link_pose, *local_pose) if father_pose_type == 'mujoco' else Pose(*link_pose)
        link.set_joint_properties(
            joint_type, np.array([range]),
            father_pose, Pose(*local_pose),
            friction, damping
        )
        link.set_collision_groups(contype, conaffinity, 0, 0)
    return link


def distance(pose1, pose2):
    cc = pose1.inv() *  pose2
    return np.linalg.norm(cc.p), transforms3d.quaternions.quat2axangle(cc.q)[1]



class CameraRender:
    def __init__(self, scene: sapien.Scene, name, width: int, height: int, fov=0.618, near=1, far=100):
        self.camera: sapien.CameraEntity = scene.add_camera(
            name, width, height, fov, near, far
        )

    def render(self):
        self.camera.take_picture()
        return (self.camera.get_color_rgba()[:, :, :3] * 255).astype(np.uint8)

    def set_camera_position(self, x, y, z):
        #pose = self.actor.pose
        pose = self.camera.get_pose()
        pose = Pose(np.array([x, y, z]), pose.q)
        self.camera.set_local_pose(pose)

    def set_camera_rotation(self, yaw, pitch):
        # yaw += 1.57 * 4
        quat = transforms3d.euler.euler2quat(0, -pitch, yaw)
        pose = self.camera.get_pose()
        pose = Pose(pose.p, quat)
        self.camera.set_local_pose(pose)

    def set_current_scene(self, scene):
        pass


def check_contact(i, a, b):
    #print(i.actor1.name, i.actor2.name)
    actor1 = i.actor1.name
    actor2 = i.actor2.name
    return (actor1==a and actor2==b) or (actor2==a and actor1==b)


def state_vector(objects):
    return np.concatenate([np.array(obj.pack()) for obj in objects])

def load_state_vector(objects, vec):
    l = 0
    for obj in objects:
        r = l + len(obj.pack())
        obj.unpack(vec[l:r])
        l = r
    assert l == len(vec)
