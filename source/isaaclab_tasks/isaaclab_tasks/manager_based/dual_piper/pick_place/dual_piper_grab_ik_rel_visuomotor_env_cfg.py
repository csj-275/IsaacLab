# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dual-arm Piper grab environment with visuomotor (image) observations."""

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from . import mdp
from .grab_env_cfg import _dual_obs_terms, _R_LEFT, _R_RIGHT, _SHARED_SPECS
from .dual_piper_grab_ik_rel_env_cfg import (
    DualPiperGrabIkRelEnvCfg,
    SubtaskCfg,
    TerminationsCfg,
    _MUG_SPECS,
)

# D435 camera intrinsic matrices
PIPER_D435_COLOR_INTRINSIC_640X480 = [
    605.519378662109, 0.0, 320.0,
    0.0, 605.519378662109, 240.0,
    0.0, 0.0, 1.0,
]

PIPER_D435_COLOR_INTRINSIC_1280X720 = [
    1211.038757324218, 0.0, 640.0,
    0.0, 908.279067993164, 360.0,
    0.0, 0.0, 1.0,
]


def _image_cpu(env, sensor_cfg, data_type, convert_perspective_to_orthogonal=False, normalize=True):
    """Wrapper that returns CPU tensors to avoid GPU memory bloat."""
    return mdp.image(env, sensor_cfg, data_type, convert_perspective_to_orthogonal, normalize).cpu()


# Uncomment to enable lighting/texture randomization:
# from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR
# from isaaclab_tasks.manager_based.dual_piper.pick_place.mdp import dual_piper_grab_events


# ---------------------------------------------------------------------------
# EventCfg with advanced randomization (commented — uncomment as needed)
# ---------------------------------------------------------------------------
#
# @configclass
# class EventCfg(grab_joint_pos_env_cfg.EventCfg):
#     \"\"\"Configuration for events.\"\"\"
#
#     # Full randomization: lighting + sky texture (indoor scenes)
#     # randomize_light = EventTerm(
#     #     func=dual_piper_grab_events.randomize_scene_lighting_domelight,
#     #     mode="reset",
#     #     params={
#     #         "intensity_range": (1500.0, 10000.0),
#     #         "color_variation": 0.4,
#     #         "textures": [
#     #             f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/autoshop_01_4k.hdr",
#     #             f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/carpentry_shop_01_4k.hdr",
#     #             f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/hospital_room_4k.hdr",
#     #             f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/hotel_room_4k.hdr",
#     #             f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/old_bus_depot_4k.hdr",
#     #             f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/small_empty_house_4k.hdr",
#     #             f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/surgery_4k.hdr",
#     #             f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/wooden_garage_4k.hdr",
#     #         ],
#     #         "default_intensity": 3000.0,
#     #         "default_color": (0.75, 0.75, 0.75),
#     #         "default_texture": "",
#     #     },
#     # )
#
#     # Lighting-only randomization: varies intensity + color, sky stays fixed
#     # randomize_light = EventTerm(
#     #     func=dual_piper_grab_events.randomize_scene_lighting_domelight,
#     #     mode="reset",
#     #     params={
#     #         "intensity_range": (500.0, 3000.0),
#     #         "color_variation": 0.4,
#     #         "textures": [f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/entrance_hall_4k.hdr"],
#     #         "default_intensity": 1500.0,
#     #         "default_color": (0.75, 0.75, 0.75),
#     #         "default_texture": "",
#     #     },
#     # )
#
#     # Table visual material randomization
#     # randomize_table_visual_material = EventTerm(
#     #     func=dual_piper_grab_events.randomize_visual_texture_material,
#     #     mode="reset",
#     #     params={
#     #         "asset_cfg": SceneEntityCfg("table"),
#     #         "textures": [
#     #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Bamboo_Planks/Bamboo_Planks_BaseColor.png",
#     #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Cherry/Cherry_BaseColor.png",
#     #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Mahogany_Planks/Mahogany_Planks_BaseColor.png",
#     #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Oak/Oak_BaseColor.png",
#     #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Timber/Timber_BaseColor.png",
#     #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Timber_Cladding/Timber_Cladding_BaseColor.png",
#     #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Walnut_Planks/Walnut_Planks_BaseColor.png",
#     #             # Metals
#     #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Steel_Stainless/Steel_Stainless_BaseColor.png",
#     #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Brass/Brass_BaseColor.png",
#     #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Bronze/Bronze_BaseColor.png",
#     #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Copper/Copper_BaseColor.png",
#     #         ],
#     #         "default_texture": "",
#     #     },
#     # )
#
#     # Warehouse RectLight: dim initial + slow large drift during episode
#     # randomize_warehouse_lights = EventTerm(
#     #     func=dual_piper_grab_events.randomize_warehouse_light_intensity,
#     #     mode="reset",
#     #     params={"intensity_range": (200.0, 1200.0), "delta_range": (0.0, 0.0)},
#     # )
#     # randomize_warehouse_lights_dynamic = EventTerm(
#     #     func=dual_piper_grab_events.randomize_warehouse_light_intensity,
#     #     mode="interval",
#     #     interval_range_s=(1.5, 4.0),
#     #     params={"delta_range": (-600.0, 600.0)},
#     # )


# ---------------------------------------------------------------------------
# Policy observations (state + image)
# ---------------------------------------------------------------------------

@configclass
class ObservationsCfg:
    """Observation specifications for dual-arm visuomotor task."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group (state + image)."""

        # 7D joint observations
        joint_pos_7d_left = ObsTerm(
            func=mdp.joint_pos_with_gripper_7d, params={"robot_cfg": _R_LEFT},
        )
        joint_pos_7d_right = ObsTerm(
            func=mdp.joint_pos_with_gripper_7d, params={"robot_cfg": _R_RIGHT},
        )
        joint_pos_target_7d_left = ObsTerm(
            func=mdp.joint_pos_target_7d, params={"robot_cfg": _R_LEFT},
        )
        joint_pos_target_7d_right = ObsTerm(
            func=mdp.joint_pos_target_7d, params={"robot_cfg": _R_RIGHT},
        )

        # Image observations (CPU)
        table_cam = ObsTerm(
            func=_image_cpu,
            params={"sensor_cfg": SceneEntityCfg("table_cam"), "data_type": "rgb", "normalize": False},
        )
        wrist_cam_left = ObsTerm(
            func=_image_cpu,
            params={"sensor_cfg": SceneEntityCfg("wrist_cam_left"), "data_type": "rgb", "normalize": False},
        )
        wrist_cam_right = ObsTerm(
            func=_image_cpu,
            params={"sensor_cfg": SceneEntityCfg("wrist_cam_right"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

            # Auto-generate _left / _right EE + object observations
            for k, v in _dual_obs_terms(_SHARED_SPECS).items():
                setattr(self, k, v)

            from .grab_env_cfg import _OBJECT_SPECS
            for k, v in _dual_obs_terms(_OBJECT_SPECS + _MUG_SPECS).items():
                setattr(self, k, v)

    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


# ---------------------------------------------------------------------------
# Environment config
# ---------------------------------------------------------------------------

@configclass
class DualPiperGrabIkRelVisuomotorEnvCfg(DualPiperGrabIkRelEnvCfg):
    """Dual-arm Piper visuomotor pick-and-place environment.

    Inherits robot setup, IK actions, gripper config, events, and terminations
    from :class:`DualPiperGrabIkRelEnvCfg`. Adds cameras, color randomization,
    and high-quality rendering.
    """

    observations: ObservationsCfg = ObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    eval_mode = False
    eval_type = None

    def __post_init__(self):
        super().__post_init__()

        # ---- Cameras -----------------------------------------------------------
        # Wrist camera — left arm
        self.scene.wrist_cam_left = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot_Left/camera_link/wrist_cam",
            update_period=1 / 30,
            width=1280,
            height=720,
            data_types=["rgb"],
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

        # Wrist camera — right arm
        self.scene.wrist_cam_right = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot_Right/camera_link/wrist_cam",
            update_period=1 / 30,
            width=1280,
            height=720,
            data_types=["rgb"],
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

        # Table camera (global overhead)
        self.scene.table_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/table_cam",
            update_period=1 / 30,
            width=1280,
            height=720,
            data_types=["rgb"],
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

        # ---- Override events: swap IK's pose ranges for visuomotor variant -----
        del self.events.randomize_cube_and_mug_and_box_poses

        self.events.randomize_cube_and_mug_and_box_poses = EventTerm(
            func=mdp.randomize_object_pose,
            mode="reset",
            params={
                "pose_ranges": [
                    {"x": (0.2, 0.35), "y": (-0.05, 0.15), "z": (0.0203, 0.0203), "yaw": (-0.785, 0.785)},  # cube
                    {"x": (0.2, 0.35), "y": (-0.05, 0.15), "z": (0.0000, 0.0000), "yaw": (-0.785, 0.785)},  # mug
                    {"x": (0.1, 0.3), "y": (0.05, 0.35), "z": (0.0000, 0.0000), "yaw": (-0.785, 0.785)},  # box
                ],
                "min_separation": 0.12,
                "asset_cfgs": [SceneEntityCfg("object_1"), SceneEntityCfg("mug"), SceneEntityCfg("box")],
            },
        )

        # ---- Visual color randomization ---------------------------------------
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

        # ---- Rendering settings (maximum quality) ------------------------------
        self.num_rerenders_on_reset = 3
        self.sim.render.antialiasing_mode = "DLAA"
        # Lighting & reflections
        self.sim.render.enable_translucency = True
        self.sim.render.enable_reflections = True
        self.sim.render.enable_global_illumination = True
        self.sim.render.enable_ambient_occlusion = True
        self.sim.render.enable_direct_lighting = True
        self.sim.render.enable_shadows = True
        # Sampling & denoising
        self.sim.render.samples_per_pixel = 32
        self.sim.render.enable_dl_denoiser = True
        self.sim.render.dlss_mode = 3
        # self.sim.render.dlss_mode = 0  # disable DLSS for max performance

        self.image_obs_list = ["table_cam", "wrist_cam_left", "wrist_cam_right"]
        self.sim.render_interval = 2
