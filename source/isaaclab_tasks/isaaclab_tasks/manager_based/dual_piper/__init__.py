# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# Cart-pole placeholder (template)
gym.register(
    id="Isaac-Dual-Piper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dual_piper_env_cfg:DualPiperEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

# Dual-arm Piper grab — joint position control
gym.register(
    id="Isaac-Dual-Piper-Grab-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_place.dual_piper_grab_joint_pos_env_cfg:DualPiperGrabJointPosEnvCfg",
    },
)

# Dual-arm Piper grab — IK relative pose control
gym.register(
    id="Isaac-Dual-Piper-Grab-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_place.dual_piper_grab_ik_rel_env_cfg:DualPiperGrabIkRelEnvCfg",
    },
)

# Dual-arm Piper grab — IK relative pose + visuomotor
gym.register(
    id="Isaac-Dual-Piper-Grab-IK-Rel-Visuomotor-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_place.dual_piper_grab_ik_rel_visuomotor_env_cfg:DualPiperGrabIkRelVisuomotorEnvCfg",
    },
)
