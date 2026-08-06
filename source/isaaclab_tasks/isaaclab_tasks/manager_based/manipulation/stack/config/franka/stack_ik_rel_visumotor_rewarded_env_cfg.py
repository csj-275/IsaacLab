# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.sensors import TiledCameraCfg


##
# Pre-defined configs
##

from isaaclab_tasks.manager_based.manipulation.stack.config.franka.stack_ik_rel_visuomotor_env_cfg import (
    FrankaCubeStackVisuomotorEnvCfg,
)  # isort:skip
from isaaclab_tasks.manager_based.manipulation.stack import mdp  # isort:skip


##
# Scene definition
##


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    success = RewTerm(func=mdp.cubes_stacked, weight=20.0)


##
# Environment configuration
##


@configclass
class FrankaCubeStackVisuomotorRewardedEnvCfg(FrankaCubeStackVisuomotorEnvCfg):
    rewards: RewardsCfg = RewardsCfg()  # add reward terms

    def __post_init__(self):
        super().__post_init__()
        self.sim.render_interval = (
            5  # set render interval as the same as the decimation interval
        )

        # Set cameras
        # Set wrist camera
        self.scene.wrist_cam = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
            update_period=0.0,
            height=200,
            width=200,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 2),
            ),
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.13, 0.0, -0.15),
                rot=(-0.70614, 0.03701, 0.03701, -0.70614),
                convention="ros",
            ),
        )

        # Set table view camera
        self.scene.table_cam = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/table_cam",
            update_period=0.0,
            height=200,
            width=200,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 2),
            ),
            offset=TiledCameraCfg.OffsetCfg(
                pos=(1.0, 0.0, 0.4),
                rot=(0.35355, -0.61237, -0.61237, 0.35355),
                convention="ros",
            ),
        )
