# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
    Author: csj
    Created: 26-04-27
    Modified: 26-04-27
    Description: Configuration the env for the piper robot with IK.
"""

from isaaclab.utils import configclass
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg, DifferentialIKControllerCfg
from . import piper_env_cfg

@configclass
class PiperEnvCfg(piper_env_cfg.PiperEnvCfg):
    """Configuration for the piper environment."""
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
    
        self.actions.arm_actions = DifferentialInverseKinematicsActionCfg(
                asset_name="robot",
                joint_names=["joint[1-6]"],
                body_name="link6",
                controller=DifferentialIKControllerCfg(
                    command_type="pose", 
                    use_relative_mode=True,
                    ik_method="dls"
                ),
                scale=0.5,
                body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                    pos=(0.12, 0.0, 0.0)
                ),
            )
