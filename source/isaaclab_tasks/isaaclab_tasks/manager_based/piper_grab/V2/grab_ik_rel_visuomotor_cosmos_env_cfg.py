# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.piper_grab import mdp
from isaaclab_tasks.manager_based.piper_grab.V2.grab_ik_rel_env_cfg import SubtaskCfg

from . import grab_ik_rel_visuomotor_env_cfg
from ..grab_ik_rel_visuomotor_env_cfg import PIPER_D435_COLOR_INTRINSIC_1280X720


@configclass
class ObservationsCfg:
    """Observation specifications for V2 cosmos task."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with cosmos-specific image observations."""

        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object = ObsTerm(func=mdp.object_obs)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)
        object_1_positions = ObsTerm(
            func=mdp.object_poses_in_base_frame,
            params={"object_cfg": SceneEntityCfg("object_1"), "return_key": "pos"},
        )
        object_1_orientations = ObsTerm(
            func=mdp.object_poses_in_base_frame,
            params={"object_cfg": SceneEntityCfg("object_1"), "return_key": "quat"},
        )
        box_positions = ObsTerm(
            func=mdp.object_poses_in_base_frame, params={"object_cfg": SceneEntityCfg("box"), "return_key": "pos"}
        )
        box_orientations = ObsTerm(
            func=mdp.object_poses_in_base_frame,
            params={"object_cfg": SceneEntityCfg("box"), "return_key": "quat"},
        )
        mug_positions = ObsTerm(
            func=mdp.object_poses_in_base_frame, params={"object_cfg": SceneEntityCfg("mug"), "return_key": "pos"}
        )
        mug_orientations = ObsTerm(
            func=mdp.object_poses_in_base_frame,
            params={"object_cfg": SceneEntityCfg("mug"), "return_key": "quat"},
        )
        table_cam = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("table_cam"), "data_type": "rgb", "normalize": False}
        )
        wrist_cam = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb", "normalize": False}
        )
        table_cam_depth = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("table_cam"),
                "data_type": "distance_to_image_plane",
                "normalize": True,
            },
        )
        wrist_cam_depth = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("wrist_cam"),
                "data_type": "distance_to_image_plane",
                "normalize": True,
            },
        )
        # Cosmos-specific observations
        # table_cam_segmentation = ObsTerm(
        #     func=mdp.image,
        #     params={"sensor_cfg": SceneEntityCfg("table_cam"), "data_type": "semantic_segmentation", "normalize": True},
        # )
        # table_cam_normals = ObsTerm(
        #     func=mdp.image,
        #     params={"sensor_cfg": SceneEntityCfg("table_cam"), "data_type": "normals", "normalize": True},
        # )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class PiperGrabVisuomotorCosmosEnvCfg(grab_ik_rel_visuomotor_env_cfg.PiperGrabVisuomotorEnvCfg):
    """Configuration for V2 two-stage pick-and-place cosmos environment."""

    observations: ObservationsCfg = ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.sim.render.dome_light_upper_lower_strategy = 4

        # Rendering quality settings
        self.sim.render.rendering_mode = "quality"
        self.sim.render.enable_translucency = True
        self.sim.render.enable_reflections = True
        self.sim.render.enable_global_illumination = True
        self.sim.render.enable_ambient_occlusion = True
        self.sim.render.enable_direct_lighting = True
        self.sim.render.samples_per_pixel = 4
        self.sim.render.enable_dl_denoiser = True
        self.num_rerenders_on_reset = 1
        self.sim.render.dlss_mode = 2
        self.sim.render.antialiasing_mode = "DLAA"
        SEMANTIC_MAPPING = {
            "class:object_1": (120, 230, 255, 255),
            "class:box": (55, 255, 139, 255),
            "class:table": (255, 237, 218, 255),
            "class:ground": (100, 100, 100, 255),
            "class:robot": (204, 110, 248, 255),
            "class:mug": (255, 200, 100, 255),
            "class:UNLABELLED": (150, 150, 150, 255),
            "class:BACKGROUND": (200, 200, 200, 255),
        }

        # Wrist camera with cosmos data types
        self.scene.wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/camera_link/wrist_cam",
            update_period=1 / 30,
            height=720,
            width=1280,
            data_types=["rgb", "semantic_segmentation", "normals", "distance_to_image_plane"],
            colorize_semantic_segmentation=True,
            semantic_segmentation_mapping=SEMANTIC_MAPPING,
            spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
                intrinsic_matrix=PIPER_D435_COLOR_INTRINSIC_1280X720,
                width=1280,
                height=720,
                clipping_range=(0.01, 2.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.0),
                rot=(0.4739127, -0.5647872, 0.5175321, -0.434261),
                convention="ros",
            ),
        )

        # Table camera with cosmos data types
        self.scene.table_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/table_cam",
            update_period=1 / 30,
            height=720,
            width=1280,
            data_types=["rgb", "semantic_segmentation", "normals", "distance_to_image_plane"],
            colorize_semantic_segmentation=True,
            semantic_segmentation_mapping=SEMANTIC_MAPPING,
            spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
                intrinsic_matrix=PIPER_D435_COLOR_INTRINSIC_1280X720,
                width=1280,
                height=720,
                clipping_range=(0.01, 2.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.0, 0.30, 0.5),
                rot=(-0.1269, 0.6437, -0.7150, 0.2414),
                convention="ros",
            ),
        )

        self.image_obs_list = ["table_cam", "wrist_cam"]
