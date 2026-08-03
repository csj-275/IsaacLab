# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_a_is_into_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_a_cfg: SceneEntityCfg = SceneEntityCfg("object_a"),
    object_b_cfg: SceneEntityCfg = SceneEntityCfg("object_b"),
    xy_threshold: float = 0.08,
    height_threshold: float = 0.05,
    height_diff: float = 0.0,
    gripper_threshold: float | None = None,  # None → use env.cfg.gripper_threshold
) -> torch.Tensor:
    """Check if an object a is put into another object b by the specified robot."""

    robot: Articulation = env.scene[robot_cfg.name]
    object_a: RigidObject = env.scene[object_a_cfg.name]
    object_b: RigidObject = env.scene[object_b_cfg.name]

    # check object a is into object b
    pos_diff = object_a.data.root_pos_w - object_b.data.root_pos_w
    height_dist = torch.linalg.vector_norm(pos_diff[:, 2:], dim=1)
    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)

    xy_ok = xy_dist < xy_threshold
    h_ok = (height_dist - height_diff) < height_threshold
    success_pose = torch.logical_and(xy_ok, h_ok)

    # Check gripper positions
    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        surface_gripper = env.scene.surface_grippers["surface_gripper"]
        suction_cup_status = surface_gripper.state.view(-1)  # 1: closed, 0: closing, -1: open
        suction_cup_is_open = (suction_cup_status == -1).to(torch.float32)
        success = torch.logical_and(suction_cup_is_open, success_pose)
    else:
        if hasattr(env.cfg, "gripper_joint_names"):
            gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
            assert len(gripper_joint_ids) == 2, "Terminations only support parallel gripper for now"

            if hasattr(env.cfg, "gripper_open_vals"):
                open_vals = env.cfg.gripper_open_vals
            else:
                open_vals = [env.cfg.gripper_open_val] * len(gripper_joint_ids)

            _grip_th = gripper_threshold if gripper_threshold is not None else env.cfg.gripper_threshold
            g0 = torch.abs(robot.data.joint_pos[:, gripper_joint_ids[0]] - open_vals[0])
            g1 = torch.abs(robot.data.joint_pos[:, gripper_joint_ids[1]] - open_vals[1])
            g0_ok = g0 < _grip_th
            g1_ok = g1 < _grip_th
            gripper_ok = torch.logical_and(g0_ok, g1_ok)
            success = torch.logical_and(success_pose, gripper_ok)

            # if not torch.any(success):
            #     import logging
            #     _log = logging.getLogger(__name__)
            #     _log.warning(
            #         f"[object_a_is_into_b] xy={xy_dist[0].item():.4f} th={xy_threshold} ok={xy_ok[0].item()}, "
            #         f"h={height_dist[0].item():.4f} th={height_threshold} ok={h_ok[0].item()}, "
            #         f"gripper joints={robot.data.joint_pos[:, gripper_joint_ids][0].tolist()} "
            #         f"open_vals={open_vals} th={_grip_th} "
            #         f"g0={g0[0].item():.4f} g1={g1[0].item():.4f}"
            #     )
        else:
            raise ValueError("No gripper_joint_names found in environment config")

    return success


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
    b_into_c = torch.logical_and(xy_dist_b < xy_threshold, (height_dist_b - height_diff) < height_threshold)

    # Check gripper is open (released the last object)
    if check_gripper_open and hasattr(env.cfg, "gripper_joint_names"):
        from isaaclab_tasks.manager_based.dual_piper.pick_place.mdp.observations import _gripper_open_targets

        gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
        open_targets = _gripper_open_targets(env, robot, gripper_joint_ids)
        gripper_pos = robot.data.joint_pos[:, gripper_joint_ids]
        gripper_open = torch.all(torch.abs(gripper_pos - open_targets) < env.cfg.gripper_threshold, dim=1)
    else:
        gripper_open = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    return torch.logical_and(torch.logical_and(a_into_c, b_into_c), gripper_open)



# def task_done(
#     env: ManagerBasedRLEnv,
#     robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     object_1_cfg: SceneEntityCfg = SceneEntityCfg("object_1"),
#     object_2_cfg: SceneEntityCfg = SceneEntityCfg("object_2"),
#     box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
#     xy_threshold: float = 0.03,  # xy_distance_threshold
#     height_threshold: float = 0.04,  # height_distance_threshold
#     height_diff: float = 0.0,  # expected height_diff
# ) -> torch.Tensor:
#     robot: Articulation = env.scene[robot_cfg.name]
#     object_1: RigidObject = env.scene[object_1_cfg.name]
#     object_2: RigidObject = env.scene[object_2_cfg.name]
#     box: RigidObject = env.scene[box_cfg.name]