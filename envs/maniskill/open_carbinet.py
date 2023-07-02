from collections import OrderedDict

import numpy as np
import sapien.core as sapien
import trimesh
from sapien.core import Pose
from scipy.spatial import distance as sdist

from mani_skill2.agents.robots.mobile_panda import MobilePandaSingleArm
from mani_skill2.utils.common import np_random, random_choice
from mani_skill2.utils.geometry import angle_distance, transform_points
from mani_skill2.utils.trimesh_utils import (
    get_articulation_meshes,
    get_visual_body_meshes,
    merge_meshes,
)

from mani_skill2.envs.ms1.base_env import MS1BaseEnv


def clip_and_normalize(x, a_min, a_max=None):
    if a_max is None:
        a_max = np.abs(a_min)
        a_min = -a_max
    return (np.clip(x, a_min, a_max) - a_min) / (a_max - a_min)


class OpenCabinetEnv(MS1BaseEnv):
    agent: MobilePandaSingleArm
    MAX_DOF = 8

    def __init__(
        self, *args, 
        fixed_target_link_idx: int = None, 
        target_links=None, 

        # x, y, z, doors' dof, and contact distance?
        obs_dim = 8,
        reward_type='dense',

        **kwargs
    ):
        # The index in target links (not all links)
        self._fixed_target_link_idx = fixed_target_link_idx
        self.my_target_links = target_links
        self._cache_bboxes = {}

        self.obs_dim = obs_dim
        self.reward = reward_type

        super().__init__(*args, **kwargs)

    def _setup_cameras(self):
        super()._setup_cameras()
        
        self.render_camera.set_local_pose(
            Pose(p=[-1.5, 0, 1.5], q=[0.9238795, 0, 0.3826834, 0])
        )

    # -------------------------------------------------------------------------- #
    # Reconfigure
    # -------------------------------------------------------------------------- #
    def _load_articulations(self):
        urdf_config = dict(
            material=dict(static_friction=1, dynamic_friction=1, restitution=0),
        )
        scale = self.model_info["scale"]
        self.cabinet = self._load_partnet_mobility(
            fix_root_link=True, scale=scale, urdf_config=urdf_config
        )
        self.cabinet.set_name(self.model_id)

        assert self.cabinet.dof <= self.MAX_DOF, self.cabinet.dof
        self._set_cabinet_handles()
        self._ignore_collision()

        if self._reward_mode == "dense":
            self.cabinet.set_pose(Pose())
            self._set_cabinet_handles_mesh()
            self._compute_handles_grasp_poses()

    def _set_cabinet_handles(self, joint_type: str):
        self.target_links = []
        self.target_joints = []
        self.target_handles = []

        # NOTE(jigu): links and their parent joints.
        for link, joint in zip(self.cabinet.get_links(), self.cabinet.get_joints()):
            if joint.type != joint_type:
                continue
            handles = []
            for visual_body in link.get_visual_bodies():
                if "handle" not in visual_body.name:
                    continue
                handles.append(visual_body)
            if len(handles) > 0:
                self.target_links.append(link)
                self.target_joints.append(joint)
                self.target_handles.append(handles)

    def _set_cabinet_handles_mesh(self):
        self.target_handles_mesh = []

        for handle_visuals in self.target_handles:
            meshes = []
            for visual_body in handle_visuals:
                meshes.extend(get_visual_body_meshes(visual_body))
            handle_mesh = merge_meshes(meshes)
            self.target_handles_mesh.append(handle_mesh)

    def _compute_grasp_poses(self, mesh: trimesh.Trimesh, pose: sapien.Pose):
        # NOTE(jigu): only for axis-aligned horizontal and vertical cases
        mesh2: trimesh.Trimesh = mesh.copy()
        # Assume the cabinet is axis-aligned canonically
        mesh2.apply_transform(pose.to_transformation_matrix())
        extents = mesh2.extents
        if extents[1] > extents[2]:  # horizontal handle
            closing = np.array([0, 0, 1])
        else:  # vertical handle
            closing = np.array([0, 1, 0])
        approaching = [1, 0, 0]
        grasp_poses = [
            self.agent.build_grasp_pose(approaching, closing, [0, 0, 0]),
            self.agent.build_grasp_pose(approaching, -closing, [0, 0, 0]),
        ]

        pose_inv = pose.inv()
        grasp_poses = [pose_inv * x for x in grasp_poses]

        return grasp_poses

    def _compute_handles_grasp_poses(self):
        self.target_handles_grasp_poses = []
        for i in range(len(self.target_handles)):
            link = self.target_links[i]
            mesh = self.target_handles_mesh[i]
            grasp_poses = self._compute_grasp_poses(mesh, link.pose)
            self.target_handles_grasp_poses.append(grasp_poses)

    def _ignore_collision(self):
        """Ignore collision within the articulation to avoid impact from imperfect collision shapes."""
        # The legacy version only ignores collision of child links of active joints.
        for link in self.cabinet.get_links():
            for s in link.get_collision_shapes():
                g0, g1, g2, g3 = s.get_collision_groups()
                s.set_collision_groups(g0, g1, g2 | 1 << 31, g3)

    def _load_agent(self):
        self.agent = MobilePandaSingleArm(
            self._scene, self._control_freq, self._control_mode
        )

    # -------------------------------------------------------------------------- #
    # Reset
    # -------------------------------------------------------------------------- #
    def reset(self, seed=None, reconfigure=False, model_id=None):
        return super().reset(seed=seed, reconfigure=reconfigure, model_id=model_id)

    def _initialize_task(self):
        self._initialize_cabinet()
        self._initialize_robot()
        self._set_target_link()
        self._set_joint_physical_parameters()

    def _compute_cabinet_bbox(self):
        mesh = merge_meshes(get_articulation_meshes(self.cabinet))
        return mesh.bounds  # [2, 3]

    def _initialize_cabinet(self):
        # Set joint positions to lower bounds
        qlimits = self.cabinet.get_qlimits()  # [N, 2]
        assert not np.isinf(qlimits).any(), qlimits
        qpos = np.ascontiguousarray(qlimits[:, 0])
        # NOTE(jigu): must use a contiguous array for `set_qpos`
        self.cabinet.set_qpos(qpos)

        # If the scale can change, caching does not work.
        bounds = self._cache_bboxes.get(self.model_id, None)
        if bounds is None:
            # The bound is computed based on current qpos.
            # NOTE(jigu): Make sure the box is computed at a canoncial pose.
            self.cabinet.set_pose(Pose())
            bounds = self._compute_cabinet_bbox()
            self._cache_bboxes[self.model_id] = bounds
        self.cabinet.set_pose(Pose([0, 0, -bounds[0, 2]]))
        
    def _initialize_robot(self):
        # Base position
        # The forward direction of cabinets is -x.
        center = np.array([0, 0.8 - 0.3]) # closer to link1
        dist = self._episode_rng.uniform(1.6, 1.8)
        theta = self._episode_rng.uniform(0.9 * np.pi, 1.1 * np.pi)
        # theta = 1.
        theta = np.pi
        direction = np.array([np.cos(theta), np.sin(theta)])
        xy = center + direction * dist

        # Base orientation
        noise_ori = self._episode_rng.uniform(-0.05 * np.pi, 0.05 * np.pi)
        ori = (theta - np.pi) + noise_ori

        h = 1e-4
        arm_qpos = np.array([0, 0, 0, -1.5, 0, 3, 0.78, 0.02, 0.02])

        qpos = np.hstack([xy, ori, h, arm_qpos])
        self.agent.reset(qpos)

    def _set_joint_physical_parameters(self):
        for joint in self.cabinet.get_active_joints():
            joint.set_friction(self._episode_rng.uniform(0.05, 0.15))
            joint.set_drive_property(
                stiffness=0, damping=self._episode_rng.uniform(5, 20)
            )

    def _set_target_link(self):
        # if self._fixed_target_link_idx is None:
        #     indices = np.arange(len(self.target_links))
        #     self.target_link_idx = random_choice(indices, rng=self._episode_rng)
        # else:
        #     self.target_link_idx = self._fixed_target_link_idx
        # assert self.target_link_idx < len(self.target_links), self.target_link_idx
        self.target_handle_mesh = {}
        self.target_handle_pcd = {} 
        self.target_handle_sdf = {} 
        self.target_link = {}
        self.target_joint = {}
        self.target_joint_idx_q = {}
        self.target_qpos = {}
        for target_link_idx in range(len(self.target_links)):

            self.target_link[target_link_idx]: sapien.Link = self.target_links[target_link_idx]
            target_joint: sapien.Joint = self.target_joints[target_link_idx]
            self.target_joint[target_link_idx] = target_joint
            # The index in active joints
            self.target_joint_idx_q[target_link_idx] = self.cabinet.get_active_joints().index(
                target_joint
            )

            qmin, qmax = target_joint.get_limits()[0]
            self.target_qpos[target_link_idx] = qmin + (qmax - qmin) * 0.9

            # target_indicator[self.target_joint_idx_q] = 1

            # Cache handle point cloud
            if self._reward_mode == "dense":
                self._set_target_handle_info(target_link_idx)

        # One-hot indicator for which link is target
        self.target_indicator = np.zeros(self.MAX_DOF, np.float32)

    def _set_target_handle_info(self, target_link_idx):
        self.target_handle_mesh[target_link_idx] = self.target_handles_mesh[target_link_idx]
        with np_random(self._episode_seed):
            # TODO(jigu): make sure how many points are needed
            self.target_handle_pcd[target_link_idx] = self.target_handle_mesh[target_link_idx].sample(100)
        self.target_handle_sdf[target_link_idx] = trimesh.proximity.ProximityQuery(
            self.target_handle_mesh[target_link_idx]
        )

    # -------------------------------------------------------------------------- #
    # Success metric and shaped reward
    # -------------------------------------------------------------------------- #
    @property
    def link_qpos(self):
        return self.cabinet.get_qpos()[self.target_joint_idx_q]

    @property
    def link_qvel(self):
        return self.cabinet.get_qvel()[self.target_joint_idx_q]

    def evaluate(self, **kwargs) -> dict:
        infos = {}

        qpos = self.cabinet.get_qpos()

        for target_link_idx in range(len(self.target_links)):
            vel_norm = np.linalg.norm(self.target_link[target_link_idx].velocity)
            ang_vel_norm = np.linalg.norm(self.target_link[target_link_idx].angular_velocity)
            link_qpos = qpos[target_link_idx]

            flags = dict(
                open_enough=link_qpos >= self.target_qpos[target_link_idx],
            )

            infos[target_link_idx] = dict(
                success=all(flags.values()),
                **flags,
                link_vel_norm=vel_norm,
                link_ang_vel_norm=ang_vel_norm,
                link_qpos=link_qpos
            )
        return infos

    def compute_dense_reward(self, *args, info: dict, **kwargs):
        # -------------------------------------------------------------------------- #
        # The end-effector should be close to the target pose
        # -------------------------------------------------------------------------- #
        ee_to_handles = []
        reward_open = []
        total_success = 0

        target_links = [0, 1] if self.my_target_links is None else self.my_target_links

        for target_link_id in target_links: # just grasp the first ..
            handle_pose = self.target_link[target_link_id].pose
            # ee_pose = self.agent.hand.pose

            # Position
            ee_coords = self.agent.get_ee_coords_sample()  # [2, 10, 3]
            handle_pcd = transform_points(
                handle_pose.to_transformation_matrix(), self.target_handle_pcd[target_link_id]
            )
            # trimesh.PointCloud(handle_pcd).show()
            disp_ee_to_handle = sdist.cdist(ee_coords.reshape(-1, 3), handle_pcd)
            dist_ee_to_handle = disp_ee_to_handle.reshape(2, -1).min(-1)  # [2]
            # reward_ee_to_handle = -
            #reward += reward_ee_to_handle
            ee_to_handles.append(dist_ee_to_handle.mean() * 2)
            link_qpos = info[target_link_id]["link_qpos"]

            if target_link_id == 0:
                reward_qpos = clip_and_normalize(link_qpos, 0, self.target_qpos[target_link_id])
                if self.reward != 'dense':
                    reward_qpos = reward_qpos > 0.8
                    total_success += reward_qpos > 0 #info[target_link_id]["success"]
                else:
                    total_success += info[target_link_id]["success"] * 2.
                    reward_qpos += info[target_link_id]["success"] * 2.

                reward_open.append(reward_qpos)

            if target_link_id == 1:
                reward_qpos = clip_and_normalize(link_qpos, 0, self.target_qpos[target_link_id])
                if self.reward != 'dense':
                    reward_qpos = reward_qpos > 0.8
                    total_success += reward_qpos > 0 # info[target_link_id]["success"]
                else:
                    total_success += info[target_link_id]["success"]
                    reward_qpos += info[target_link_id]["success"]

                reward_open.append(reward_qpos)


        if self.reward != 'dense':
            info.update(success=int(total_success > 1))
        else:
            info.update(success=total_success)
            info.update(metric=dict(door0=info[0]['success']))
        
        if self.reward == 'dense':
            return np.sum(reward_open) - np.min(ee_to_handles)
        elif self.reward == 'sparse':
            return int(np.sum(reward_open) > 1)
        elif self.reward == 'partial':
            return np.sum(reward_open)
        else:
            raise NotImplementedError

    # -------------------------------------------------------------------------- #
    # Observations
    # -------------------------------------------------------------------------- #
    def _get_obs_priviledged(self):
        obs = super()._get_obs_priviledged()
        obs["target_indicator"] = self.target_indicator
        return obs

    def _get_task_articulations(self):
        # The maximum DoF is 6 in our data.
        return [(self.cabinet, 8)]

    def get_done(self, info: dict, **kwargs):
        return False # never stop early ..

    def _render_traj_rgb(self, traj, occ_val=False, history=None, verbose=True, **kwargs):
        from ..utils import extract_obs_from_tarj
        obs = (extract_obs_from_tarj(traj))[..., :2]

        output = {
            'state': obs,
            'background': {},
            'history': {},
            'image': {},
            'metric': {}
        }

        return output


    def wrap_obs(self, obs):
        ee_xyz = self.agent.get_ee_coords().mean(axis=0)
        assert ee_xyz.shape == (3,)
        return np.concatenate([ee_xyz, obs])

    def step(self, action):
        #raise NotImplementedError
        obs, reward, done, info = super().step(action)
        return self.wrap_obs(obs), reward, done, info
        

    def reset(self, seed=None, reconfigure=False):
        obs = super().reset(seed, reconfigure)
        if self.obs_mode == "rgbd":
            return obs
        return self.wrap_obs(obs)


# @register_gym_env(name="OpenCabinetDoor-v1", max_episode_steps=200)
# class OpenCabinetDoorEnv(OpenCabinetEnv):
#     DEFAULT_MODEL_JSON = (
#         "{ASSET_DIR}/partnet_mobility/meta/info_cabinet_door_train.json"
#     )

#     def _set_cabinet_handles(self):
#         super()._set_cabinet_handles("revolute")

class OpenCabinetDoorEnv(OpenCabinetEnv):
    DEFAULT_MODEL_JSON = (
        "{ASSET_DIR}/partnet_mobility/meta/info_cabinet_door_train.json"
    )

    def _set_cabinet_handles(self):
        super()._set_cabinet_handles("revolute")


# @register_gym_env(name="OpenCabinetDrawer-v1", max_episode_steps=200)
# class OpenCabinetDrawerEnv(OpenCabinetEnv):
#     DEFAULT_MODEL_JSON = (
#         "{ASSET_DIR}/partnet_mobility/meta/info_cabinet_drawer_train.json"
#     )

#     def _set_cabinet_handles(self):
#         super()._set_cabinet_handles("prismatic")
