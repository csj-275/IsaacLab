# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""V1-A variant: minimal state observation (actions only, 7-dim) for compatibility with
LeRobot checkpoints trained on datasets where observation.state = [7]."""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg

from isaaclab_tasks.manager_based.piper_grab import mdp
from isaaclab_tasks.manager_based.piper_grab.V1.grab_ik_rel_env_cfg import SubtaskCfg

from .grab_ik_rel_visuomotor_env_cfg import (
    PiperGrabVisuomotorEnvCfg,
    _image_cpu,
)


class ObservationsCfg_V1_A:
    """Minimal observation: actions (7d) + images only. No joint/eef/gripper state."""

    class PolicyCfg(ObsGroup):
        """Observations for policy group — actions-only state."""

        actions = ObsTerm(func=mdp.last_action)
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
    """V1-A: same env as V1 but with minimal state observation (7-dim actions only)."""

    observations: ObservationsCfg_V1_A = ObservationsCfg_V1_A()

    def __post_init__(self):
        super().__post_init__()
