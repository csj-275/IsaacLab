# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""V1-A variant: joint-position action (7D) + inherits observations from base V1.

The base V1 ObservationsCfg already provides joint_pos_7d (current joint positions),
table_cam, wrist_cam, and other terms — exactly what the ACT policy needs as input.
V1-A only overrides the action space to use direct joint position targets instead
of IK delta poses.
"""
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg

from .grab_ik_rel_visuomotor_env_cfg import PiperGrabVisuomotorEnvCfg

from isaaclab_tasks.manager_based.piper_grab import mdp


class PiperGrabVisuomotorEnvCfg_V1_A(PiperGrabVisuomotorEnvCfg):
    """V1-A: joint-position action, inherits V1 observations (joint_pos_7d + cameras)."""
    def __post_init__(self):
        super().__post_init__()

        # 绝对关节位置
        self.actions.arm_action = JointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint[1-6]"],
            scale=1.0,
            use_default_offset=False,
        )

        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
                    asset_name="robot",
                    joint_names=["joint7", "joint8"],
                    open_command_expr={"joint7": 0.05, "joint8":-0.05},
                    close_command_expr={"joint7": -0.05, "joint8":0.05},
        )
