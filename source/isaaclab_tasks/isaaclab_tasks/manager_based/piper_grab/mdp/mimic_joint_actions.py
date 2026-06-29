# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom action term that enforces joint8 = -joint7 via software-level mimic."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.envs.mdp.actions import actions_cfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class MimicBinaryJointPositionAction(BinaryJointPositionAction):
    """Binary joint position action with software-level mimic coupling.

    Controls joint7 via a rate-limited PD target. Directly writes joint8's
    physics position to the mirror of joint7's *actual* position each frame,
    guaranteeing |joint7| == |joint8| even when contact forces push the
    fingers asymmetrically.
    """

    cfg: "MimicBinaryJointPositionActionCfg"

    def __init__(self, cfg: "MimicBinaryJointPositionActionCfg", env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        # Resolve the slave (mimic) joint
        self._mimic_joint_ids, self._mimic_joint_names = self._asset.find_joints(self.cfg.mimic_joint_names)
        if len(self._mimic_joint_ids) == 0:
            raise ValueError(f"No mimic joints found for pattern: {self.cfg.mimic_joint_names}")
        # Smooth target buffer — initialise to the open position so the
        # gripper doesn't snap on the first frame.
        init_target = torch.tensor(
            [float(self.cfg.open_command_expr[k]) for k in self._joint_names],
            device=self.device,
        ).repeat(self.num_envs, 1)
        self._smooth_target = init_target

    def process_actions(self, actions: torch.Tensor):
        # Let the parent compute the desired (instant) binary target
        super().process_actions(actions)
        desired = self._processed_actions.clone()

        # Rate-limit: move smooth target towards desired at limited speed
        max_delta = self.cfg.max_speed_per_step
        diff = desired - self._smooth_target
        diff = torch.clamp(diff, -max_delta, max_delta)
        self._smooth_target = self._smooth_target + diff

        # Replace the instantaneous target with the smoothed one
        self._processed_actions = self._smooth_target

    def apply_actions(self):
        # Drive joint7 via the (smoothed) PD target
        super().apply_actions()
        # Mirror joint8's physics state to joint7.
        # We write position + velocity directly, then also set the PD target
        # to the same position so the PD controller produces zero force.
        # This prevents reaction impulses from propagating to the arm.
        master_pos = self._asset.data.joint_pos[:, self._joint_ids[0]]
        master_vel = self._asset.data.joint_vel[:, self._joint_ids[0]]
        for mimic_id in self._mimic_joint_ids:
            slave_pos = self.cfg.mimic_multiplier * master_pos + self.cfg.mimic_offset
            slave_vel = self.cfg.mimic_multiplier * master_vel
            slave_pos_2d = slave_pos.unsqueeze(1)
            slave_vel_2d = slave_vel.unsqueeze(1)
            self._asset.write_joint_state_to_sim(
                position=slave_pos_2d,
                velocity=slave_vel_2d,
                joint_ids=[mimic_id],
            )
            # Sync PD target to the written position → PD error = 0 → no force
            self._asset.set_joint_position_target(slave_pos_2d, joint_ids=[mimic_id])


@configclass
class MimicBinaryJointPositionActionCfg(actions_cfg.BinaryJointPositionActionCfg):
    """Configuration for the mimic binary joint position action term."""

    class_type: type[MimicBinaryJointPositionAction] = MimicBinaryJointPositionAction

    mimic_joint_names: list[str] = ["joint8"]
    """Joint name pattern for the slave (mimic) joint."""

    mimic_multiplier: float = -1.0
    """Multiplier applied to the master target to compute the slave target."""

    mimic_offset: float = 0.0
    """Offset added to the slave target after multiplication."""

    max_speed_per_step: float = 0.005
    """Maximum target change per simulation step (joint units/step).

    At 60 fps the full open→close stroke (0.10 units) takes about
    0.10 / 0.005 = 20 steps ≈ 0.33 s.  Increase for faster motion,
    decrease for slower.  Set to a very large value (e.g. 1.0) to
    effectively disable the rate limiter.
    """
