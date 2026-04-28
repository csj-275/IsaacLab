# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# no checked

def return_target_joint_pos(env):
    """return the target pos of joint (num_envs, 2)"""
    return env.target_joint_pos

def joint_pos_error_l2(env, asset_cfg: SceneEntityCfg):
    """返回 joint1, joint2 当前角度与期望角度的 L2 平方误差"""
    asset = env.scene[asset_cfg.name]
    # 获取关节索引（支持正则，如 "joint[1-2]"）
    indices, _ = asset.find_joints(asset_cfg.joint_names)
    current_pos = asset.data.joint_pos[:, indices]  # (num_envs, 2)
    target_pos = env.target_joint_pos               # (num_envs, 2)
    error = current_pos - target_pos
    return torch.sum(error ** 2, dim=1)

def randomize_initial_state_and_target(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["joint[1-2]", "wheel_joint"]),
    pos_range: tuple = (-0.3, 0.3),
    vel_range: tuple = (-0.5, 0.5),
    target_range: tuple = (-0.8, 0.8),
):
    """随机重置关节位置、速度，并生成新的 joint1/joint2 目标位置"""
    asset = env.scene[asset_cfg.name]
    # 重置所有关节（joint1, joint2, wheel_joint）的位置和速度
    joints, _ = asset.find_joints(asset_cfg.joint_names)
    num_joints = len(joints)
    # 位置随机
    new_pos = torch.rand(env.num_envs, num_joints, device=env.device)
    new_pos = new_pos * (pos_range[1] - pos_range[0]) + pos_range[0]
    asset.write_joint_state_to_sim(new_pos, joint_ids=joints)
    # 速度随机
    new_vel = torch.rand(env.num_envs, num_joints, device=env.device)
    new_vel = new_vel * (vel_range[1] - vel_range[0]) + vel_range[0]
    asset.write_joint_velocity_to_sim(new_vel, joint_ids=joints)
    # 生成新的期望位置（仅 joint1 和 joint2）
    target = torch.rand(env.num_envs, 2, device=env.device)
    target = target * (target_range[1] - target_range[0]) + target_range[0]
    env.target_joint_pos = target

# def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
#     """Penalize joint position deviation from a target value."""
#     # extract the used quantities (to enable type-hinting)
#     asset: Articulation = env.scene[asset_cfg.name]
#     # wrap the joint positions to (-pi, pi)
#     joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
#     # compute the reward
#     return torch.sum(torch.square(joint_pos - target), dim=1)
