# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""V1-A variant: joint-position action (7D) + joint-target state (7D).

Matches training dataset: action=joint_pos_7d (actual), state=joint_pos_target_7d (target).
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg, BinaryJointPositionActionCfg

from isaaclab_tasks.manager_based.piper_grab.V1 import mdp as V1_mdp
from isaaclab_tasks.manager_based.piper_grab.V1.grab_ik_rel_env_cfg import SubtaskCfg

from .grab_ik_rel_visuomotor_env_cfg import (
    PiperGrabVisuomotorEnvCfg,
    _image_cpu,
)


class ObservationsCfg_V1_A:
    """Observation: joint_pos_target_7d (state) + images. No IK/actions state."""

    class PolicyCfg(ObsGroup):
        """Observations for policy group — joint target only (7D)."""

        state = ObsTerm(func=V1_mdp.joint_pos_target_7d)  # 7D commanded joint targets
        table_cam = ObsTerm(
            func=_image_cpu,
            params={"sensor_cfg": SceneEntityCfg("table_cam"), "data_type": "rgb", "normalize": False},
        )
        wrist_cam = ObsTerm(
            func=_image_cpu,
            params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


class PiperGrabVisuomotorEnvCfg_V1_A(PiperGrabVisuomotorEnvCfg):
    """V1-A: joint-position action + joint-target state for joint-trained policy."""

    observations: ObservationsCfg_V1_A = ObservationsCfg_V1_A()

    def __post_init__(self):
        super().__post_init__()

        # Force override observations (parent may add extra state terms)
        self.observations = ObservationsCfg_V1_A()

        # Override arm action: IK delta pose → joint position targets (6D)
        self.actions.arm_action = JointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint[1-6]"],
            scale=1,
        )

        # Override gripper: binary (1D scalar → open/close)
        self.actions.gripper_action = BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint7", "joint8"],
            open_command_expr={"joint7": 0.05, "joint8": -0.05},
            close_command_expr={"joint7": -0.05, "joint8": 0.05},
        )

        # Gripper status
        self.gripper_joint_names = ["joint7", "joint8"]
        self.gripper_open_vals = [0.05, -0.05]
        self.gripper_threshold = 0.01
