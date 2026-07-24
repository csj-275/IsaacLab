# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Test environment config for ACT policy
"""
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from .grab_ik_rel_visuomotor_env_cfg import PiperGrabVisuomotorEnvCfg

class PiperGrabVisuomotorEnvCfg_V1_A(PiperGrabVisuomotorEnvCfg):
    """action: joint-position action"""
    def __post_init__(self):
        super().__post_init__()

        self.actions.arm_action = JointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint[1-6]"],
            scale=1.0,
            use_default_offset=False,
        )

