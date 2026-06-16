# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

import torch

from isaaclab_mimic.envs.piper_grab_ik_abs_mimic_env import PiperGrabIKAbsMimicEnv


class PiperGrabIKAbsV1MimicEnv(PiperGrabIKAbsMimicEnv):
    """Isaac Lab Mimic environment wrapper for Piper Grab IK Abs V1 task.

    Extends the IK-abs base with V1-specific subtask termination signals:
    grasp_1 (cube grasped), placed_1 (cube in box), grasp_2 (mug grasped).
    """

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Get subtask termination signals for V1 task.

        Returns grasp_1 (cube grasped), placed_1 (cube in box), grasp_2 (mug grasped).

        Args:
            env_ids: Environment indices to get the termination signals for. If None, all envs are considered.

        Returns:
            A dictionary of termination signal flags (False or True) for each subtask.
        """
        if env_ids is None:
            env_ids = slice(None)

        signals = dict()
        subtask_terms = self.obs_buf["subtask_terms"]
        signals["grasp_1"] = subtask_terms["grasp_1"][env_ids]
        signals["placed_1"] = subtask_terms["placed_1"][env_ids]
        signals["grasp_2"] = subtask_terms["grasp_2"][env_ids]
        return signals
