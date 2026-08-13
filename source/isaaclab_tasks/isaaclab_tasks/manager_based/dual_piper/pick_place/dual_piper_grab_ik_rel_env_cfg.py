# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dual-arm Piper grab environment with differential IK (relative pose) control."""

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from .dual_piper_grab_joint_pos_env_cfg import DualPiperGrabJointPosEnvCfg

# ---------------------------------------------------------------------------
# Environment config
# ---------------------------------------------------------------------------

@configclass
class DualPiperGrabIkRelEnvCfg(DualPiperGrabJointPosEnvCfg):
    """Dual-arm Piper two-stage pick-and-place with differential IK control."""

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
