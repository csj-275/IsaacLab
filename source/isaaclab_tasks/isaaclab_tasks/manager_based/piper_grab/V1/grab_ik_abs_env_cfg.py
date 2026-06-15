# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""V1 IK-absolute config — same as IK-rel but with absolute IK controller."""

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.piper_grab.V1.grab_ik_rel_env_cfg import PiperGrabEnvCfg as IKRelEnvCfg


@configclass
class PiperGrabEnvCfg(IKRelEnvCfg):
    """V1 IK-absolute: same as IK-rel, but arm_action uses absolute IK."""

    def __post_init__(self):
        super().__post_init__()

        # Override to absolute IK
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["joint[1-6]"],
            body_name="link6",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            scale=1.0,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        )
