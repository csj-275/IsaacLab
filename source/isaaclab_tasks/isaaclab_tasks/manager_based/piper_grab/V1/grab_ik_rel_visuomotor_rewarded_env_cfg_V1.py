# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rewarded visuomotor environment config for ACT policy evaluation with RL reward."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from .grab_ik_rel_visuomotor_env_cfg_V1_A import PiperGrabVisuomotorEnvCfg_V1_A
from . import mdp as V1_mdp


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    success = RewTerm(func=V1_mdp.objects_a_and_b_are_into_c, weight=20.0)


@configclass
class PiperGrabVisuomotorRewardedEnvCfg_V1(PiperGrabVisuomotorEnvCfg_V1_A):
    """Configuration for V1 two-stage pick-and-place visuomotor environment with reward."""

    rewards: RewardsCfg = RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.sim.render_interval = 5