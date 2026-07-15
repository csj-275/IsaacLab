# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.piper_grab import mdp

from .grab_ik_rel_env_cfg import ObservationsCfg as V1ObservationsCfg
from .grab_ik_rel_env_cfg import SubtaskCfg as V1SubtaskCfg
from . import grab_ik_rel_env_cfg

##
# Pre-defined configs
##


@configclass
class ObservationsCfg(V1ObservationsCfg):
    """Inherits PolicyCfg and subtask_terms from V1; adds RGB camera, extends subtask with placed_2."""

    @configclass
    class SubtaskCfg(V1SubtaskCfg):
        """Inherits grasp_1, placed_1, grasp_2 from V1; adds placed_2 (mug -> box)."""

        placed_2 = ObsTerm(
            func=mdp.object_a_is_into_b,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "object_a_cfg": SceneEntityCfg("mug"),
                "object_b_cfg": SceneEntityCfg("box"),
                "xy_threshold": 0.07,
                "height_diff": 0.0,
                "height_threshold": 0.1,
                "gripper_threshold": 0.03,
            },
        )

    @configclass
    class RGBCameraPolicyCfg(ObsGroup):
        """Observations for policy group with RGB images."""

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    rgb_camera: RGBCameraPolicyCfg = RGBCameraPolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class PiperGrabSkillgenEnvCfg(grab_ik_rel_env_cfg.PiperGrabEnvCfg):
    """Configuration for V1 skillgen IK-rel environment.

    Inherits all scene, event, termination, and device configuration from V1's
    PiperGrabEnvCfg. Only overrides observations with the skillgen-specific config
    (Policy + RGBCamera + Subtask groups).
    """

    observations: ObservationsCfg = ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.observations = ObservationsCfg()
