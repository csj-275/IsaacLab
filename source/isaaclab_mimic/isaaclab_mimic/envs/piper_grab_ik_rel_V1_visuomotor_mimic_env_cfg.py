# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.piper_grab.V1.grab_ik_rel_visuomotor_env_cfg import PiperGrabVisuomotorEnvCfg


@configclass
class PiperGrabIKRelV1VisuomotorMimicEnvCfg(PiperGrabVisuomotorEnvCfg, MimicEnvCfg):
    """Mimic environment config for Piper Grab IK Rel Visuomotor V1 two-stage task."""

    def __post_init__(self):
        super().__post_init__()

        self.datagen_config.name = "isaac_lab_piper_grab_V1_visuomotor_D0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = True
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_transform_first_robot_pose = True
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.generation_relative = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 1

        subtask_configs = []
        subtask_configs.append(
            SubTaskConfig(
                object_ref="object_1",
                subtask_term_signal="grasp_1",
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.00,
                num_interpolation_steps=10,
                num_fixed_steps=5,
                apply_noise_during_interpolation=False,
                description="Grasp the cube",
                next_subtask_description="Place the cube into the box",
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                object_ref="box",
                subtask_term_signal="placed_1",
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.00,
                num_interpolation_steps=10,
                num_fixed_steps=5,
                apply_noise_during_interpolation=False,
                description="Place cube into box",
                next_subtask_description="Grasp the mug",
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                object_ref="mug",
                subtask_term_signal="grasp_2",
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.00,
                num_interpolation_steps=10,
                num_fixed_steps=5,
                apply_noise_during_interpolation=False,
                description="Grasp the mug",
                next_subtask_description="Place the mug into the box",
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                object_ref="box",
                subtask_term_signal=None,
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.00,
                num_interpolation_steps=10,
                num_fixed_steps=300,
                apply_noise_during_interpolation=False,
            )
        )

        self.subtask_configs["piper"] = subtask_configs
