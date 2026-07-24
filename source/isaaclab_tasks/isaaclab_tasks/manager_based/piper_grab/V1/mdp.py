# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _gripper_1d(env: ManagerBasedRLEnv, robot: Articulation) -> torch.Tensor:
    """Return 1D gripper scalar from joint7, joint8: ``(joint7 - joint8) / 2``."""
    if hasattr(env.cfg, "gripper_joint_names"):
        gripper_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
        grip_values = robot.data.joint_pos[:, gripper_ids]  # (N, 2): joint7, joint8
        return (grip_values[:, 0] - grip_values[:, 1]) / 2.0  # (N,)
    return torch.zeros(env.num_envs, device=env.device)


def joint_pos_with_gripper_7d(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return 7D joint positions: joint1-6 + 1D gripper (observed, post-step)."""
    robot: Articulation = env.scene[robot_cfg.name]
    joint_ids, _ = robot.find_joints("joint[1-6]")
    arm_pos = robot.data.joint_pos[:, joint_ids]  # (N, 6)
    gripper = _gripper_1d(env, robot)
    return torch.cat([arm_pos, gripper.unsqueeze(1)], dim=1)  # (N, 7)


def joint_pos_target_7d(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return 7D joint position TARGETS: joint1-6 + 1D gripper target.

    These are the ACTUAL commanded joint positions (action_K), computed by
    the IK controller from the IK delta pose action. Use this as the recorded
    ``action`` column for joint-position training.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    all_ids, _ = robot.find_joints("joint[1-8]")
    all_targets = robot.data.joint_pos_target[:, all_ids]  # (N, 8): joint1-8 targets
    arm_target = all_targets[:, :6]  # (N, 6): joint1-6
    # Gripper target from joint7/joint8 targets: (joint7_target - joint8_target) / 2
    gripper = (all_targets[:, 6] - all_targets[:, 7]) / 2.0  # (N,)
    return torch.cat([arm_target, gripper.unsqueeze(1)], dim=1)  # (N, 7)


def objects_a_and_b_are_into_c(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_a_cfg: SceneEntityCfg = SceneEntityCfg("object_a"),
    object_b_cfg: SceneEntityCfg = SceneEntityCfg("object_b"),
    object_c_cfg: SceneEntityCfg = SceneEntityCfg("object_c"),
    xy_threshold: float = 0.07,
    height_threshold: float = 0.1,
    height_diff: float = 0.0,
    check_gripper_open: bool = True,
) -> torch.Tensor:
    """Check if both object_a AND object_b are placed into object_c, and gripper is open.

    Args:
        env: The environment.
        robot_cfg: The robot scene entity config.
        object_a_cfg: First object to check.
        object_b_cfg: Second object to check.
        object_c_cfg: The container object.
        xy_threshold: Maximum horizontal distance between each object and container.
        height_threshold: Maximum vertical distance threshold.
        height_diff: Expected height difference between each object and container.
    """
    object_a: RigidObject = env.scene[object_a_cfg.name]
    object_b: RigidObject = env.scene[object_b_cfg.name]
    object_c: RigidObject = env.scene[object_c_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    # Check object_a is in object_c
    pos_diff_a = object_a.data.root_pos_w - object_c.data.root_pos_w
    xy_dist_a = torch.linalg.vector_norm(pos_diff_a[:, :2], dim=1)
    height_dist_a = pos_diff_a[:, 2]
    a_into_c = torch.logical_and(xy_dist_a < xy_threshold, (height_dist_a - height_diff) < height_threshold)

    # Check object_b is in object_c
    pos_diff_b = object_b.data.root_pos_w - object_c.data.root_pos_w
    xy_dist_b = torch.linalg.vector_norm(pos_diff_b[:, :2], dim=1)
    height_dist_b = pos_diff_b[:, 2]
    xy_ok_b = xy_dist_b < xy_threshold
    h_ok_b = (height_dist_b - height_diff) < height_threshold
    b_into_c = torch.logical_and(xy_ok_b, h_ok_b)
    
    # Check gripper is open (released the last object)
    if check_gripper_open and hasattr(env.cfg, "gripper_joint_names"):
        from isaaclab_tasks.manager_based.piper_grab.mdp.observations import _gripper_open_targets

        gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
        open_targets = _gripper_open_targets(env, robot, gripper_joint_ids)
        gripper_pos = robot.data.joint_pos[:, gripper_joint_ids]
        gripper_open = torch.all(torch.abs(gripper_pos - open_targets) < env.cfg.gripper_threshold, dim=1)
    else:
        # fallback: assume gripper is open if no gripper config
        gripper_open = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    # if torch.logical_and(torch.logical_and(a_into_c, b_into_c), ~gripper_open).any():
    #     print(
    #         f"[objects_a_and_b_are_into_c] BOTH objects in box but GRIPPER NOT OPEN! "
    #         f"gripper_pos={gripper_pos[0].tolist()}, open_targets={open_targets.tolist()}, "
    #         f"threshold={env.cfg.gripper_threshold}",
    #         flush=True,
    #     )
    # print(
    #     f"[objects_a_and_b_are_into_c] a_into_c={a_into_c[0].item()}, b_into_c={b_into_c[0].item()}, "
    #     f"gripper_open={gripper_open[0].item()}, "
    #     f"gripper_pos={gripper_pos[0].tolist()}, th={env.cfg.gripper_threshold}",
    #     flush=True,
    # )

    return torch.logical_and(torch.logical_and(a_into_c, b_into_c), gripper_open)
