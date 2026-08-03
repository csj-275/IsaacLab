# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Concrete dual-arm Piper grab environment: joint-position control."""

from isaaclab.assets import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

from isaaclab_assets.robots.piper import PIPER_STANDARD_WITH_GRIPPER_HIGH_PD_CFG  # isort: skip

from . import mdp
from .grab_env_cfg import GrabEnvCfg


# ---------------------------------------------------------------------------
# Helper: build a FrameTransformerCfg for one arm
# ---------------------------------------------------------------------------

def _make_ee_frame(robot_prim: str, side: str) -> FrameTransformerCfg:
    """Create a FrameTransformerCfg for the *side* arm (``"left"`` or ``"right"``)."""
    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg.prim_path = f"/Visuals/FrameTransformer_{side}"

    return FrameTransformerCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{robot_prim}/arm_base",
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{robot_prim}/link6",
                name=f"end_effector_{side}",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{robot_prim}/link7",
                name=f"tool_leftfinger_{side}",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.135)),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{robot_prim}/link8",
                name=f"tool_rightfinger_{side}",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.135)),
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

@configclass
class EventCfg:
    """Configuration for dual-arm reset events."""

    # -- left arm -----------------------------------------------------------
    init_left_arm_pose = EventTerm(
        func=mdp.set_default_joint_pose,
        mode="reset",
        params={
            "default_pose": [0.0, 1.0, -0.6, 0.0, 1.35, 0.0, 0.05, -0.05],
            "asset_cfg": SceneEntityCfg("robot_left"),
        },
    )

    randomize_left_joint_state = EventTerm(
        func=mdp.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.0,
            "asset_cfg": SceneEntityCfg("robot_left"),
        },
    )

    # -- right arm ----------------------------------------------------------
    init_right_arm_pose = EventTerm(
        func=mdp.set_default_joint_pose,
        mode="reset",
        params={
            "default_pose": [0.0, 1.0, -0.6, 0.0, 1.35, 0.0, 0.05, -0.05],
            "asset_cfg": SceneEntityCfg("robot_right"),
        },
    )

    randomize_right_joint_state = EventTerm(
        func=mdp.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.0,
            "asset_cfg": SceneEntityCfg("robot_right"),
        },
    )

    # -- objects ------------------------------------------------------------
    randomize_cube_positions = EventTerm(
        func=mdp.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {"x": (0.25, 0.5), "y": (-0.2, 0.2), "z": (0.0203, 0.0203), "yaw": (-1.0, 1.0)},
            "min_separation": 0.1,
            "asset_cfgs": [SceneEntityCfg("object_1")],
        },
    )

    randomize_box_positions = EventTerm(
        func=mdp.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {"x": (0.05, 0.3), "y": (0.15, 0.35), "z": (0.2203, 0.2203), "yaw": (-1.0, 1.0)},
            "min_separation": 0.1,
            "asset_cfgs": [SceneEntityCfg("box")],
        },
    )


# ---------------------------------------------------------------------------
# Environment config
# ---------------------------------------------------------------------------

@configclass
class DualPiperGrabJointPosEnvCfg(GrabEnvCfg):
    """Dual-arm Piper grab environment with joint-position control."""

    def __post_init__(self):
        super().__post_init__()

        # -- Shared gripper parameters (identical Piper arms) --------------
        self.gripper_joint_names = ["joint7", "joint8"]
        self.gripper_open_vals = [0.05, -0.05]
        self.gripper_threshold = 0.01

        # -- Events --------------------------------------------------------
        self.events = EventCfg()

        # ------------------------------------------------------------------
        # Left arm
        # ------------------------------------------------------------------
        self.scene.robot_left = PIPER_STANDARD_WITH_GRIPPER_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot_Left"
        )
        self.scene.robot_left.init_state = ArticulationCfg.InitialStateCfg(
            pos=(0.0, -0.25, 0.0),
        )
        self.scene.robot_left.spawn.semantic_tags = [("class", "robot")]

        self.scene.ee_frame_left = _make_ee_frame("Robot_Left", "left")

        self.actions.arm_action_left = mdp.JointPositionActionCfg(
            asset_name="robot_left", joint_names=["joint[1-6]"], scale=1.0
        )

        self.actions.gripper_action_left = mdp.MimicBinaryJointPositionActionCfg(
            asset_name="robot_left",
            joint_names=["joint7"],
            open_command_expr={"joint7": 0.05},
            close_command_expr={"joint7": -0.05},
            mimic_joint_names=["joint8"],
            mimic_multiplier=-1.0,
            max_speed_per_step=0.01,
        )

        # ------------------------------------------------------------------
        # Right arm
        # ------------------------------------------------------------------
        self.scene.robot_right = PIPER_STANDARD_WITH_GRIPPER_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot_Right"
        )
        self.scene.robot_right.init_state = ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.25, 0.0),
        )
        self.scene.robot_right.spawn.semantic_tags = [("class", "robot")]

        self.scene.ee_frame_right = _make_ee_frame("Robot_Right", "right")

        self.actions.arm_action_right = mdp.JointPositionActionCfg(
            asset_name="robot_right", joint_names=["joint[1-6]"], scale=1.0
        )

        self.actions.gripper_action_right = mdp.MimicBinaryJointPositionActionCfg(
            asset_name="robot_right",
            joint_names=["joint7"],
            open_command_expr={"joint7": 0.05},
            close_command_expr={"joint7": -0.05},
            mimic_joint_names=["joint8"],
            mimic_multiplier=-1.0,
            max_speed_per_step=0.01,
        )

        # ------------------------------------------------------------------
        # Scene semantics
        # ------------------------------------------------------------------
        self.scene.table.spawn.semantic_tags = [("class", "table")]
        self.scene.plane.semantic_tags = [("class", "ground")]
