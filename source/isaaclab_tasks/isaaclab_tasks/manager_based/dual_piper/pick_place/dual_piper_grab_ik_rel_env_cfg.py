# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dual-arm Piper grab environment with differential IK (relative pose) control."""

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim.spawners.materials.visual_materials_cfg import PreviewSurfaceCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.piper import PIPER_STANDARD_WITH_GRIPPER_HIGH_PD_CFG  # isort: skip

from . import mdp
from .dual_piper_grab_joint_pos_env_cfg import DualPiperGrabJointPosEnvCfg
from .grab_env_cfg import _dual_obs_terms, _R_LEFT, _R_RIGHT, _EE_LEFT, _EE_RIGHT, _SHARED_SPECS

# ---- Extra object specs (mug) --------------------------------------------------
_MUG_SPECS: list[tuple[str, callable, dict, dict]] = [
    (
        "mug_positions",
        mdp.object_poses_in_base_frame,
        {"object_cfg": SceneEntityCfg("mug"), "robot_cfg": _R_LEFT, "return_key": "pos"},
        {"object_cfg": SceneEntityCfg("mug"), "robot_cfg": _R_RIGHT, "return_key": "pos"},
    ),
    (
        "mug_orientations",
        mdp.object_poses_in_base_frame,
        {"object_cfg": SceneEntityCfg("mug"), "robot_cfg": _R_LEFT, "return_key": "quat"},
        {"object_cfg": SceneEntityCfg("mug"), "robot_cfg": _R_RIGHT, "return_key": "quat"},
    ),
]


# ---------------------------------------------------------------------------
# Subtask observations
# ---------------------------------------------------------------------------

@configclass
class SubtaskCfg(ObsGroup):
    """Observations for two-stage subtask group (cube→box, mug→box)."""

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False

        for side, r_cfg, ee_cfg in [("left", _R_LEFT, _EE_LEFT), ("right", _R_RIGHT, _EE_RIGHT)]:
            # grasp cube
            setattr(self, f"grasp_cube_{side}", ObsTerm(
                func=mdp.object_grasped,
                params={
                    "robot_cfg": r_cfg,
                    "ee_frame_cfg": ee_cfg,
                    "object_cfg": SceneEntityCfg("object_1"),
                    "diff_threshold": 0.05,
                },
            ))
            # placed cube into box
            setattr(self, f"placed_cube_{side}", ObsTerm(
                func=mdp.object_a_is_into_b,
                params={
                    "robot_cfg": r_cfg,
                    "object_a_cfg": SceneEntityCfg("object_1"),
                    "object_b_cfg": SceneEntityCfg("box"),
                    "xy_threshold": 0.05,
                    "height_diff": 0.0,
                    "height_threshold": 0.05,
                    "gripper_threshold": 0.03,
                },
            ))
            # grasp mug
            setattr(self, f"grasp_mug_{side}", ObsTerm(
                func=mdp.object_grasped,
                params={
                    "robot_cfg": r_cfg,
                    "ee_frame_cfg": ee_cfg,
                    "object_cfg": SceneEntityCfg("mug"),
                    "diff_threshold": 0.05,
                },
            ))


# ---------------------------------------------------------------------------
# Policy observations
# ---------------------------------------------------------------------------

@configclass
class ObservationsCfg:
    """Observation specifications for dual-arm two-stage IK-rel task (state-only)."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        actions = ObsTerm(func=mdp.last_action)

        # Object observation (world-frame, shared; uses left ee as reference)
        object = ObsTerm(func=mdp.object_obs, params={"ee_frame_cfg": _EE_LEFT})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

            # Auto-generate _left / _right pairs for arm EE + object poses
            for k, v in _dual_obs_terms(_SHARED_SPECS + _MUG_SPECS).items():
                setattr(self, k, v)

            # object_1 + box specs (already in grab_env_cfg, but need to re-add here
            # since this PolicyCfg is a full override)
            from .grab_env_cfg import _OBJECT_SPECS
            for k, v in _dual_obs_terms(_OBJECT_SPECS).items():
                setattr(self, k, v)

    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------

@configclass
class TerminationsCfg:
    """Termination conditions for dual-arm two-stage task."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_1_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object_1")},
    )
    mug_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("mug")},
    )
    box_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("box")},
    )
    success = DoneTerm(
        func=mdp.objects_a_and_b_are_into_c,
        params={
            "robot_cfg": SceneEntityCfg("robot_left"),
            "object_a_cfg": SceneEntityCfg("object_1"),
            "object_b_cfg": SceneEntityCfg("mug"),
            "object_c_cfg": SceneEntityCfg("box"),
        },
    )


# ---------------------------------------------------------------------------
# Environment config
# ---------------------------------------------------------------------------

@configclass
class DualPiperGrabIkRelEnvCfg(DualPiperGrabJointPosEnvCfg):
    """Dual-arm Piper two-stage pick-and-place with differential IK control."""

    observations: ObservationsCfg = ObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()  # robots, ee_frames, events inherited from joint_pos

        # IK-rel actions for both arms
        ik_offset = DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.135))
        ik_controller = DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=True, ik_method="dls"
        )

        self.actions.arm_action_left = DifferentialInverseKinematicsActionCfg(
            asset_name="robot_left",
            joint_names=["joint[1-6]"],
            body_name="link6",
            controller=ik_controller,
            scale=0.5,
            body_offset=ik_offset,
        )

        self.actions.arm_action_right = DifferentialInverseKinematicsActionCfg(
            asset_name="robot_right",
            joint_names=["joint[1-6]"],
            body_name="link6",
            controller=ik_controller,
            scale=0.5,
            body_offset=ik_offset,
        )

        # Gripper config (shared)
        self.gripper_joint_names = ["joint7", "joint8"]
        self.gripper_open_vals = [0.05, -0.05]
        self.gripper_threshold = 0.015

        # ---- Override events: combined cube + mug + box randomization --------
        del self.events.randomize_cube_positions
        del self.events.randomize_box_positions

        self.events.randomize_cube_and_mug_and_box_poses = EventTerm(
            func=mdp.randomize_object_pose,
            mode="reset",
            params={
                "pose_ranges": [
                    {"x": (0.3, 0.35), "y": (0.1, 0.15), "z": (0.0203, 0.0203), "yaw": (-0.785, 0.785)},  # cube
                    {"x": (0.2, 0.25), "y": (0.1, 0.15), "z": (0.0000, 0.0000), "yaw": (-0.785, 0.785)},  # mug
                    {"x": (0.15, 0.2), "y": (0.25, 0.3), "z": (0.0000, 0.0000), "yaw": (-0.785, 0.785)},  # box
                ],
                "min_separation": 0.12,
                "asset_cfgs": [SceneEntityCfg("object_1"), SceneEntityCfg("mug"), SceneEntityCfg("box")],
            },
        )

        # ---- Visual materials -------------------------------------------------
        self.scene.mug.spawn.visual_material = PreviewSurfaceCfg(
            diffuse_color=(0.85, 0.45, 0.10),  # orange
            roughness=0.4,
            metallic=0.0,
        )
        self.scene.box.spawn.visual_material = PreviewSurfaceCfg(
            diffuse_color=(0.55, 0.35, 0.15),  # brown/cardboard
            roughness=0.5,
            metallic=0.0,
        )

        # ---- Simulation settings ----------------------------------------------
        self.sim.dt = 1 / 150
        self.decimation = 5
        self.sim.render_interval = 3
