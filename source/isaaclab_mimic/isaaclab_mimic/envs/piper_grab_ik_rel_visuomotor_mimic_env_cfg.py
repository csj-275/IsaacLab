# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.piper_grab.grab_ik_rel_visuomotor_env_cfg import (
    PiperGrabVisuomotorEnvCfg,
)


@configclass
class PiperGrabIKRelVisuomotorMimicEnvCfg(PiperGrabVisuomotorEnvCfg, MimicEnvCfg):
    """
    Isaac Lab Mimic environment config class for Piper Grab IK Rel Visuomotor env.

    This configuration supports:
    - Visuomotor observations (wrist_cam, table_cam)
    - SkillGen (cuRobo motion planning for demo-free generation)
    - MimicGen (source-demo-based data augmentation)
    """

    def __post_init__(self):
        # post init of parents
        super().__post_init__()

        # --------------------------------------------------------------------------
        # Data generation configuration
        # --------------------------------------------------------------------------
        self.datagen_config.name = "piper_grab_visuomotor_D0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = True
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.generation_relative = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 1

        # Enable SkillGen by default (uses cuRobo motion planner)
        # Can be overridden via CLI --use_skillgen flag in generate_dataset.py
        self.datagen_config.use_skillgen = True

        # --------------------------------------------------------------------------
        # Subtask configurations
        # Task: Grasp object_1 and place it into the box
        # --------------------------------------------------------------------------
        subtask_configs = []

        # Subtask 0: Grasp object_1
        subtask_configs.append(
            SubTaskConfig(
                # Object involved in this subtask
                object_ref="object_1",
                # Binary indicator in "datagen_info" signaling subtask completion
                # Matches the key in get_subtask_term_signals()
                subtask_term_signal="grasp_1",
                # Time offset range for subtask boundary randomization (MimicGen)
                subtask_term_offset_range=(10, 20),
                # Selection strategy for source subtask segment
                selection_strategy="nearest_neighbor_object",
                # Parameters for selection strategy
                selection_strategy_kwargs={"nn_k": 3},
                # Action noise amplitude
                action_noise=0.03,
                # Interpolation steps to bridge subtask segments
                num_interpolation_steps=5,
                # Fixed steps for the robot to reach the necessary pose
                num_fixed_steps=0,
                # Apply noise during interpolation
                apply_noise_during_interpolation=False,
                # SkillGen descriptions
                description="Grasp the blue cube on the table",
                next_subtask_description="Place the blue cube into the sorting bin",
            )
        )

        # Subtask 1: Place object_1 into the box (final subtask)
        subtask_configs.append(
            SubTaskConfig(
                object_ref="box",
                # End of final subtask does not need to be detected
                subtask_term_signal=None,
                # No time offsets for the final subtask
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.03,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )

        # Register subtask configs under the end-effector name
        # This key is used throughout the MimicEnv to reference the arm
        self.subtask_configs["piper_arm"] = subtask_configs
