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


def objects_a_and_b_are_into_c(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_a_cfg: SceneEntityCfg = SceneEntityCfg("object_a"),
    object_b_cfg: SceneEntityCfg = SceneEntityCfg("object_b"),
    object_c_cfg: SceneEntityCfg = SceneEntityCfg("object_c"),
    xy_threshold: float = 0.08,
    height_threshold: float = 0.10,
    height_diff: float = 0.0,
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
    if hasattr(env.cfg, "gripper_joint_names"):
        from isaaclab_tasks.manager_based.piper_grab.mdp.observations import _gripper_open_targets

        gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
        open_targets = _gripper_open_targets(env, robot, gripper_joint_ids)
        gripper_pos = robot.data.joint_pos[:, gripper_joint_ids]
        gripper_open = torch.all(torch.abs(gripper_pos - open_targets) < env.cfg.gripper_threshold, dim=1)
    else:
        # fallback: assume gripper is open if no gripper config
        gripper_open = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    # print(f"[a_into_c]: {a_into_c}, [b_into_c]: {b_into_c}, gripper_open: {gripper_open}")

    return torch.logical_and(torch.logical_and(a_into_c, b_into_c), gripper_open)
