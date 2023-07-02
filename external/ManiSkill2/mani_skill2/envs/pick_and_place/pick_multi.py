from collections import OrderedDict
from pathlib import Path
from typing import Dict, List

import numpy as np
import sapien.core as sapien
from sapien.core import Pose
from transforms3d.euler import euler2quat

from mani_skill2 import ASSET_DIR
from mani_skill2.utils.common import random_choice
from mani_skill2.utils.io_utils import load_json
from mani_skill2.utils.registration import register_gym_env
from mani_skill2.utils.sapien_utils import look_at, set_actor_visibility, vectorize_pose

from .base_env import StationaryManipulationEnv
from .pick_single import PickSingleYCBEnv, build_actor_ycb
from .stack_cube import UniformSampler


class PickMultiEnv(StationaryManipulationEnv):
    SUPPORTED_REWARD_MODES = ("sparse",)
    DEFAULT_ASSET_ROOT: str
    DEFAULT_MODEL_JSON: str

    obj: sapien.Actor  # target object

    def __init__(
        self,
        asset_root: str = None,
        model_json: str = None,
        **kwargs,
    ):
        # Root directory of object models
        if asset_root is None:
            asset_root = self.DEFAULT_ASSET_ROOT
        self._asset_root = Path(asset_root.format(ASSET_DIR=ASSET_DIR))

        # Information of object models
        if model_json is None:
            model_json = self.DEFAULT_MODEL_JSON
        model_json = self._asset_root / model_json
        self.model_db: Dict[str, Dict] = load_json(model_json)
        self.model_ids = sorted(self.model_db.keys())

        self.n_objs = 3
        self.goal_thresh = 0.025

        super().__init__(**kwargs)

    def reset(self, seed=None, reconfigure=False):
        self.set_episode_rng(seed)
        model_inds = self._episode_rng.choice(len(self.model_ids), self.n_objs)
        self._ep_model_ids = [self.model_ids[i] for i in model_inds]
        return super().reset(seed=self._episode_seed, reconfigure=True)

    def _load_actors(self):
        self._add_ground()

        self.objs: List[sapien.Actor] = []
        for model_id in self._ep_model_ids:
            model_scale = self.model_db[model_id]["scales"][0]
            obj = self._load_model(model_id, model_scale=model_scale)
            self.objs.append(obj)

        self.obj = self.objs[0]

    def set_obj(self, index):
        self.obj = self.objs[index]
        self.model_id = self._ep_model_ids[index]

    def _load_model(self, model_id, model_scale=1.0) -> sapien.Actor:
        raise NotImplementedError

    def _initialize_actors(self):
        offset = np.array([0, -0.2])
        region = [[-0.2, -0.2], [0.2, 0.2]]
        sampler = UniformSampler(region, self._episode_rng)
        for i, model_id in enumerate(self._ep_model_ids):
            model_info = self.model_db[model_id]
            bbox = model_info["bbox"]
            model_scale = self.model_db[model_id]["scales"][0]
            bbox_size = (np.array(bbox["max"]) - np.array(bbox["min"])) * model_scale
            z = -bbox["min"][-1]
            radius = np.linalg.norm(bbox_size[:2]) * 0.5 + 0.005
            xy = offset + sampler.sample(radius, 100, verbose=False)
            quat = euler2quat(0, 0, self._episode_rng.uniform(0, 2 * np.pi))
            self.objs[i].set_pose(sapien.Pose([xy[0], xy[1], z], quat))

    def _initialize_agent(self):
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
            self.agent.robot.set_pose(Pose([-0.544, 0, 0]))
        else:
            raise NotImplementedError(self.robot_uuid)

    @property
    def obj_pose(self):
        """Get the center of mass (COM) pose."""
        return self.obj.pose.transform(self.obj.cmass_local_pose)

    def evaluate(self, **kwargs):
        return dict(success=False)

    def _setup_cameras(self):
        super()._setup_cameras()
        self.render_camera.set_local_pose(look_at([0.3, 0, 1.0], [0.0, 0.0, 0.5]))


@register_gym_env("PickMultiYCB-v0", max_episode_steps=200)
class PickMultiYCBEnv(PickMultiEnv):
    DEFAULT_ASSET_ROOT = "{ASSET_DIR}/mani_skill2_ycb"
    DEFAULT_MODEL_JSON = "../pick_clutter/info_pick_clutter_v0.json"

    def _load_model(self, model_id, model_scale=1.0):
        density = self.model_db[model_id].get("density", 1000)
        obj = build_actor_ycb(
            model_id,
            self._scene,
            scale=model_scale,
            density=density,
            root_dir=self._asset_root,
        )
        obj.name = model_id
        obj.set_damping(0.1, 0.1)
        return obj
