# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR

from isaaclab.devices import DeviceBase, DevicesCfg, OpenXRDeviceCfg, Se3KeyboardCfg
from isaaclab.devices.openxr.retargeters.manipulator.gripper_retargeter import GripperRetargeterCfg
from isaaclab.devices.openxr.retargeters.manipulator.se3_rel_retargeter import Se3RelRetargeterCfg


from isaaclab_tasks.manager_based.piper_grab import mdp
from isaaclab_tasks.manager_based.piper_grab.mdp import piper_grab_events
from isaaclab_tasks.manager_based.piper_grab.V1 import mdp as V1_mdp
from isaaclab_tasks.manager_based.piper_grab.V1.grab_ik_rel_env_cfg import (
    SubtaskCfg,
    TerminationsCfg,
)

from .. import grab_joint_pos_env_cfg
from ..grab_ik_rel_visuomotor_env_cfg import PIPER_D435_COLOR_INTRINSIC_1280X720

##
# Pre-defined configs
##
from isaaclab_assets.robots.piper import PIPER_STANDARD_WITH_GRIPPER_HIGH_PD_CFG  # isort: skip
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

_MUG_USD_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../../../../../usd/bottle/bottle.usd")
)


# _MUG_USD_PATH = f"{ISAACLAB_NUCLEUS_DIR}/Objects/Mug/mug.usd"


@configclass
class EventCfg(grab_joint_pos_env_cfg.EventCfg):
    """Configuration for events."""

    # Full randomization: lighting + sky texture (indoor scenes)
    # randomize_light = EventTerm(
    #     func=piper_grab_events.randomize_scene_lighting_domelight,
    #     mode="reset",
    #     params={
    #         "intensity_range": (1500.0, 10000.0),
    #         "color_variation": 0.4,
    #         "textures": [
    #             f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/autoshop_01_4k.hdr",
    #             f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/carpentry_shop_01_4k.hdr",
    #             f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/hospital_room_4k.hdr",
    #             f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/hotel_room_4k.hdr",
    #             f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/old_bus_depot_4k.hdr",
    #             f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/small_empty_house_4k.hdr",
    #             f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/surgery_4k.hdr",
    #             f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/wooden_garage_4k.hdr",
    #         ],
    #         "default_intensity": 3000.0,
    #         "default_color": (0.75, 0.75, 0.75),
    #         "default_texture": "",
    #     },
    # )

    # Lighting-only randomization: varies intensity + color, sky stays fixed indoor
    randomize_light = EventTerm(
        func=piper_grab_events.randomize_scene_lighting_domelight,
        mode="reset",
        params={
            "intensity_range": (500.0, 3000.0),
            "color_variation": 0.4,
            "textures": [f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/entrance_hall_4k.hdr"],
            "default_intensity": 1500.0,
            "default_color": (0.75, 0.75, 0.75),
            "default_texture": "",
        },
    )

    randomize_table_visual_material = EventTerm(
        func=piper_grab_events.randomize_visual_texture_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("table"),
            "textures": [
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Ash/Ash_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Bamboo_Planks/Bamboo_Planks_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Birch/Birch_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Cherry/Cherry_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Mahogany_Planks/Mahogany_Planks_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Oak/Oak_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Plywood/Plywood_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Timber/Timber_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Timber_Cladding/Timber_Cladding_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Walnut_Planks/Walnut_Planks_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Stone/Marble/Marble_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Steel_Stainless/Steel_Stainless_BaseColor.png",
            ],
            "default_texture": (
                f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/Materials/Textures/DemoTable_TableBase_BaseColor.png"
            ),
        },
    )

    # randomize_robot_arm_visual_texture = EventTerm(
    #     func=piper_grab_events.randomize_visual_texture_material,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "textures": [
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Aluminum_Cast/Aluminum_Cast_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Aluminum_Polished/Aluminum_Polished_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Brass/Brass_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Bronze/Bronze_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Brushed_Antique_Copper/Brushed_Antique_Copper_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Cast_Metal_Silver_Vein/Cast_Metal_Silver_Vein_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Copper/Copper_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Gold/Gold_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Iron/Iron_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/RustedMetal/RustedMetal_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Silver/Silver_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Steel_Carbon/Steel_Carbon_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Steel_Stainless/Steel_Stainless_BaseColor.png",
    #         ],
    #     },
    # )


@configclass
class ObservationsCfg:
    """Observation specifications for V1 visuomotor task."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state and image values."""

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

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class PiperGrabVisuomotorEnvCfg(grab_joint_pos_env_cfg.PiperGrabEnvCfg):
    """Configuration for V1 two-stage pick-and-place visuomotor environment."""

    observations: ObservationsCfg = ObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    eval_mode = False
    eval_type = None

    def __post_init__(self):
        super().__post_init__()

        # Override events
        self.events = EventCfg()

        # Override robot to high-PD variant for IK tracking
        self.scene.robot = PIPER_STANDARD_WITH_GRIPPER_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]

        # Gripper config
        self.gripper_joint_names = ["joint7", "joint8"]
        self.gripper_open_vals = [0.05, -0.05]
        self.gripper_threshold = 0.01


        # IK-rel action
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["joint[1-6]"],
            body_name="link6",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )

        # Wrist camera
        self.scene.wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/camera_link/wrist_cam",
            update_period=1 / 30,
            width=1280,
            height=720,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
                intrinsic_matrix=PIPER_D435_COLOR_INTRINSIC_1280X720,
                width=1280,
                height=720,
                clipping_range=(0.1, 2.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.0),
                # pos=(0.0043, -0.0175, 0),
                # w x y z
                # rot=(0.5, -0.5, 0.5, -0.5),
                rot=(0.4545, -0.5417, 0.5417, -0.4545),
                # rot=(0.6123724, -0.3535534, 0.3535534, -0.6123724),
                # rot=(0.3536, -0.6124, 0.6124, -0.3536),
                convention="ros",
            ),
        )

        # Table camera
        self.scene.table_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/table_cam",
            update_period=1 / 30,
            width=1280,
            height=720,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
                intrinsic_matrix=PIPER_D435_COLOR_INTRINSIC_1280X720,
                width=1280,
                height=720,
                clipping_range=(0.1, 2.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.0, 0.30, 0.5),
                # rot=(0.9703, 0, 0.2419, 0), 
                # rot=(-0.3536, 0.6124, -0.6124, 0.3536),
                # rot=(-0.3044, 0.5272, -0.6870, 0.3967),
                # rot=(-0.1162, 0.6119, -0.7424, 0.2467),
                rot=(-0.1269, 0.6437, -0.7150, 0.2414),
                convention="ros",
            ),
        )
        # z 50-54 y 25 roty28

        # Add mug to scene
        mug_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )
        self.scene.mug = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/mug",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, -0.15, 0.0003), rot=(1, 0, 0, 0)),
            spawn=UsdFileCfg(
                usd_path=_MUG_USD_PATH,
                scale=(1, 1, 1),
                rigid_props=mug_properties,
                semantic_tags=[("class", "mug")],
            ),
        )

        # Combined cube + mug + box randomization with per-object pose ranges to avoid overlaps
        del self.events.randomize_cube_positions
        del self.events.randomize_box_positions

        self.events.randomize_cube_and_mug_and_box_poses = EventTerm(
            func=piper_grab_events.randomize_object_pose,
            mode="reset",
            params={
                "pose_ranges": [
                    {"x": (0.2, 0.35), "y": (-0.05, 0.15), "z": (0.0203, 0.0203), "yaw": (-1.0, 1.0)},  # cube
                    {"x": (0.2, 0.35), "y": (-0.05, 0.15), "z": (0.0000, 0.0000), "yaw": (-1.0, 1.0)},  # mug
                    {"x": (0.1, 0.3), "y": (0.05, 0.35), "z": (0.0000, 0.0000), "yaw": (-1.0, 1.0)},  # box
                ],
                "min_separation": 0.12,
                "asset_cfgs": [SceneEntityCfg("object_1"), SceneEntityCfg("mug"), SceneEntityCfg("box")],
            },
        )


        # Color randomization
        self.events.randomize_cube_color = EventTerm(
            func=mdp.randomize_visual_color,
            mode="reset",
            params={
                "event_name": "randomize_cube_color",
                "asset_cfg": SceneEntityCfg("object_1"),
                "colors": {"r": (0.0, 1.0), "g": (0.0, 1.0), "b": (0.0, 1.0)},
                "mesh_name": "",
            },
        )

        self.events.randomize_box_color = EventTerm(
            func=mdp.randomize_visual_color,
            mode="reset",
            params={
                "event_name": "randomize_box_color",
                "asset_cfg": SceneEntityCfg("box"),
                "colors": {"r": (0.0, 1.0), "g": (0.0, 1.0), "b": (0.0, 1.0)},
                "mesh_name": "",
            },
        )

        self.events.randomize_mug_color = EventTerm(
            func=mdp.randomize_visual_color,
            mode="reset",
            params={
                "event_name": "randomize_mug_color",
                "asset_cfg": SceneEntityCfg("mug"),
                "colors": {"r": (0.0, 1.0), "g": (0.0, 1.0), "b": (0.0, 1.0)},
                "mesh_name": "",
            },
        )

        # Rendering settings
        self.num_rerenders_on_reset = 3
        self.sim.render.antialiasing_mode = "DLAA"

        self.image_obs_list = ["table_cam", "wrist_cam"]
        self.sim.dt = 1 / 150

        self.teleop_devices = DevicesCfg(
            devices={
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        Se3RelRetargeterCfg(
                            bound_hand=DeviceBase.TrackingTarget.HAND_RIGHT,
                            zero_out_xy_rotation=True,
                            use_wrist_rotation=False,
                            use_wrist_position=True,
                            delta_pos_scale_factor=10.0,
                            delta_rot_scale_factor=10.0,
                            sim_device=self.sim.device,
                        ),
                        GripperRetargeterCfg(
                            bound_hand=DeviceBase.TrackingTarget.HAND_RIGHT, sim_device=self.sim.device
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.1,
                    rot_sensitivity=0.1,
                    sim_device=self.sim.device,
                ),
            }
        )
