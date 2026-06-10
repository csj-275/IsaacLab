# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


##
# Joint Position Control
##

gym.register(
    id="Isaac-Piper-Grab-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.grab_joint_pos_env_cfg:PiperGrabEnvCfg",
    },
)

gym.register(
    id="Isaac-Piper-Instance-Randomize-Grab-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.grab_ik_rel_instance_randomize_env_cfg:PiperGrabInstanceRandomizeEnvCfg",
    },
)

##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="Isaac-Piper-Grab-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.grab_ik_rel_env_cfg:PiperGrabEnvCfg",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_low_dim.json",
    },
)

gym.register(
    id="Isaac-Piper-Grab-IK-Rel-Instance-Randomize-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.grab_ik_rel_instance_randomize_env_cfg:PiperGrabInstanceRandomizeEnvCfg",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_low_dim.json",
    },
)

gym.register(
    id="Isaac-Piper-Grab-IK-Rel-Visuomotor-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.grab_ik_rel_visuomotor_env_cfg:PiperGrabVisuomotorEnvCfg",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_image_200.json",
    },
)

# gym.register(
#     id="Isaac-Piper-Grab-IK-Rel-Skillgen-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.grab_ik_rel_env_cfg_skillgen:PiperGrabSkillgenEnvCfg",
#         "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_low_dim.json",
#     },
#     disable_env_checker=True,
# )


gym.register(
    id="Isaac-Piper-Grab-IK-Rel-Visuomotor-Cosmos-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.grab_ik_rel_visuomotor_cosmos_env_cfg:PiperGrabVisuomotorCosmosEnvCfg",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_image_cosmos.json",
    },
    disable_env_checker=True,
)

##
# V1 Two-Stage Pick-and-Place Tasks
##

gym.register(
    id="Isaac-Piper-Grab-IK-Rel-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.V1.grab_ik_rel_env_cfg:PiperGrabEnvCfg",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_low_dim.json",
    },
)

gym.register(
    id="Isaac-Piper-Grab-IK-Rel-Visuomotor-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.V1.grab_ik_rel_visuomotor_env_cfg:PiperGrabVisuomotorEnvCfg",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_image_200.json",
    },
)

gym.register(
    id="Isaac-Piper-Grab-IK-Rel-Visuomotor-Cosmos-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.V1.grab_ik_rel_visuomotor_cosmos_env_cfg:PiperGrabVisuomotorCosmosEnvCfg",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_image_cosmos.json",
    },
)
