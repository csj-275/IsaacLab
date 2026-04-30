# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_poses_in_base_frame(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    return_key: Literal["pos", "quat", None] = None,
) -> torch.Tensor:
    """The pose of the object in the robot base frame."""
    object: RigidObject = env.scene[object_cfg.name]

    pos_object_world = object.data.root_pos_w
    quat_object_world = object.data.root_quat_w

    """The position of the robot in the world frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w

    pos_object_base, quat_object_base = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, pos_object_world, quat_object_world
    )
    if return_key == "pos":
        return pos_object_base
    elif return_key == "quat":
        return quat_object_base
    else:
        return torch.cat((pos_object_base, quat_object_base), dim=1)


def object_grasped(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    diff_threshold: float = 0.06,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """
    Check if an object is grasped by the specified robot.
    Support both surface gripper and parallel gripper.
    If contact_grasp sensor is found, check if the contact force is greater than force_threshold.
    """

    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    object_pos = object.data.root_pos_w
    end_effector_pos = ee_frame.data.target_pos_w[:, 0, :]
    pose_diff = torch.linalg.vector_norm(object_pos - end_effector_pos, dim=1)

    if "contact_grasp" in env.scene.keys() and env.scene["contact_grasp"] is not None:
        contact_force_grasp = env.scene["contact_grasp"].data.net_forces_w  # shape:(N, 2, 3) for two fingers
        contact_force_norm = torch.linalg.vector_norm(
            contact_force_grasp, dim=2
        )  # shape:(N, 2) - force magnitude per finger
        both_fingers_force_ok = torch.all(
            contact_force_norm > force_threshold, dim=1
        )  # both fingers must exceed threshold
        grasped = torch.logical_and(pose_diff < diff_threshold, both_fingers_force_ok)
    elif (
        f"contact_grasp_{object_cfg.name}" in env.scene.keys()
        and env.scene[f"contact_grasp_{object_cfg.name}"] is not None
    ):
        contact_force_object = env.scene[
            f"contact_grasp_{object_cfg.name}"
        ].data.net_forces_w  # shape:(N, 2, 3) for two fingers
        contact_force_norm = torch.linalg.vector_norm(
            contact_force_object, dim=2
        )  # shape:(N, 2) - force magnitude per finger
        both_fingers_force_ok = torch.all(
            contact_force_norm > force_threshold, dim=1
        )  # both fingers must exceed threshold
        grasped = torch.logical_and(pose_diff < diff_threshold, both_fingers_force_ok)
    else:
        grasped = (pose_diff < diff_threshold).clone().detach()

    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        surface_gripper = env.scene.surface_grippers["surface_gripper"]
        suction_cup_status = surface_gripper.state.view(-1, 1)  # 1: closed, 0: closing, -1: open
        suction_cup_is_closed = (suction_cup_status == 1).to(torch.float32)
        grasped = torch.logical_and(suction_cup_is_closed, pose_diff < diff_threshold)

    else:
        if hasattr(env.cfg, "gripper_joint_names"):
            gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
            grasped = torch.logical_and(
                grasped,
                torch.abs(torch.abs(robot.data.joint_pos[:, gripper_joint_ids[0]]) - env.cfg.gripper_open_val)
                > env.cfg.gripper_threshold,
            )
            grasped = torch.logical_and(
                grasped,
                torch.abs(torch.abs(robot.data.joint_pos[:, gripper_joint_ids[1]]) - env.cfg.gripper_open_val)
                > env.cfg.gripper_threshold,
            )
        else:
            raise ValueError("No gripper_joint_names found in environment config")

    return grasped



def ee_frame_pos(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_pos = ee_frame.data.target_pos_w[:, 0, :] - env.scene.env_origins[:, 0:3]

    return ee_frame_pos


def ee_frame_quat(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_quat = ee_frame.data.target_quat_w[:, 0, :]

    return ee_frame_quat


def gripper_pos(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Obtain the versatile gripper position of both Gripper and Suction Cup.
    """
    robot: Articulation = env.scene[robot_cfg.name]

    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        # Handle multiple surface grippers by concatenating their states
        gripper_states = []
        for gripper_name, surface_gripper in env.scene.surface_grippers.items():
            gripper_states.append(surface_gripper.state.view(-1, 1))

        if len(gripper_states) == 1:
            return gripper_states[0]
        else:
            return torch.cat(gripper_states, dim=1)

    else:
        if hasattr(env.cfg, "gripper_joint_names"):
            gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
            assert len(gripper_joint_ids) == 2, "Observation gripper_pos only support parallel gripper for now"
            finger_joint_1 = robot.data.joint_pos[:, gripper_joint_ids[0]].clone().unsqueeze(1)
            finger_joint_2 = -1 * robot.data.joint_pos[:, gripper_joint_ids[1]].clone().unsqueeze(1)
            return torch.cat((finger_joint_1, finger_joint_2), dim=1)
        else:
            raise NotImplementedError("[Error] Cannot find gripper_joint_names in the environment config")


# instance randomization

def instance_randomize_object_positions_in_world_frame(
    env: ManagerBasedRLEnv,
    object_1_cfg: SceneEntityCfg = SceneEntityCfg("object_1"),
) -> torch.Tensor:
    """The position of the cubes in the world frame."""
    if not hasattr(env, "rigid_objects_in_focus"):
        return torch.full((env.num_envs, 9), fill_value=-1)

    object_1: RigidObjectCollection = env.scene[object_1_cfg.name]

    object_1_pos_w = []
    for env_id in range(env.num_envs):
        object_1_pos_w.append(object_1.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][0], :3])
    object_1_pos_w = torch.stack(object_1_pos_w)

    return object_1_pos_w


def instance_randomize_object_orientations_in_world_frame(
    env: ManagerBasedRLEnv,
    object_1_cfg: SceneEntityCfg = SceneEntityCfg("object_1"),
) -> torch.Tensor:
    """The orientation of the cubes in the world frame."""
    if not hasattr(env, "rigid_objects_in_focus"):
        return torch.full((env.num_envs, 9), fill_value=-1)

    object_1: RigidObjectCollection = env.scene[object_1_cfg.name]

    object_1_quat_w = []
    for env_id in range(env.num_envs):
        object_1_quat_w.append(object_1.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][0], :4])
    object_1_quat_w = torch.stack(object_1_quat_w)

    return object_1_quat_w



def instance_randomize_object_obs(
    env: ManagerBasedRLEnv,
    object_1_cfg: SceneEntityCfg = SceneEntityCfg("object_1"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
):
    """
    Object observations (in world frame):
        object_1 pos,
        object_1 quat,
        gripper to object_1,
    """
    if not hasattr(env, "rigid_objects_in_focus"):
        return torch.full((env.num_envs, 9), fill_value=-1)

    object_1: RigidObjectCollection = env.scene[object_1_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    object_1_pos_w = []
    object_1_quat_w = []
    for env_id in range(env.num_envs):
        object_1_pos_w.append(object_1.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][0], :3])
        object_1_quat_w.append(object_1.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][0], :4])
    object_1_pos_w = torch.stack(object_1_pos_w)
    object_1_quat_w = torch.stack(object_1_quat_w)

    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    gripper_to_object_1 = object_1_pos_w - ee_pos_w

    return torch.cat(
        (
            object_1_pos_w - env.scene.env_origins,
            object_1_quat_w,
            gripper_to_object_1,
        ),
        dim=1,
    )


