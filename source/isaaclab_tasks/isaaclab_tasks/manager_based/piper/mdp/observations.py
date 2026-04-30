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

# done
def object_positions_in_world_frame(
    env: ManagerBasedRLEnv,
    object1_name: str,
    object2_name: str,
    object3_name: str,
)-> torch.Tensor:
    '''
    Object observations (in world frame):
        object pos,
        object quat,
    '''
    object1_pos = env.scene[object1_name].data.root_pos_w - env.scene.env_origins
    object2_pos = env.scene[object2_name].data.root_pos_w - env.scene.env_origins
    object3_pos = env.scene[object3_name].data.root_pos_w - env.scene.env_origins
    return torch.cat((object1_pos, object2_pos, object3_pos),dim=1)

# done
def instance_randomize_object_positions_in_world_frame(
    env: ManagerBasedRLEnv,
    object1_name: str,
    object2_name: str,
    object3_name: str,
) -> torch.Tensor:
    """The position of the objects in the world frame."""
    if not hasattr(env, "rigid_objects_in_focus"):
        return torch.full((env.num_envs, 9), fill_value=-1)

    object_1: RigidObjectCollection = env.scene[object1_name]
    object_2: RigidObjectCollection = env.scene[object2_name]
    object_3: RigidObjectCollection = env.scene[object3_name]

    object_1_pos_w = []
    object_2_pos_w = []
    object_3_pos_w = []
    for env_id in range(env.num_envs):
        object_1_pos_w.append(object_1.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][0], :3])
        object_2_pos_w.append(object_2.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][1], :3])
        object_3_pos_w.append(object_3.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][2], :3])
    object_1_pos_w = torch.stack(object_1_pos_w)
    object_2_pos_w = torch.stack(object_2_pos_w)
    object_3_pos_w = torch.stack(object_3_pos_w)

    return torch.cat((object_1_pos_w, object_2_pos_w, object_3_pos_w), dim=1)


def object_orientations_in_world_frame(
    env: ManagerBasedRLEnv,
    object1_name: str,
    object2_name: str,
    object3_name: str,
):
    """The orientation of the objects in the world frame."""
    object_1: RigidObject = env.scene[object1_name]
    object_2: RigidObject = env.scene[object2_name]
    object_3: RigidObject = env.scene[object3_name]

    return torch.cat((object_1.data.root_quat_w, object_2.data.root_quat_w, object_3.data.root_quat_w), dim=1)

# done
def instance_randomize_object_orientations_in_world_frame(
    env: ManagerBasedRLEnv,
    object1_name: str,
    object2_name: str,
    object3_name: str,
) -> torch.Tensor:
    """The orientation of the objects in the world frame."""
    if not hasattr(env, "rigid_objects_in_focus"):
        return torch.full((env.num_envs, 9), fill_value=-1)

    object_1: RigidObjectCollection = env.scene[object1_name]
    object_2: RigidObjectCollection = env.scene[object2_name]
    object_3: RigidObjectCollection = env.scene[object3_name]

    object_1_quat_w = []
    object_2_quat_w = []
    object_3_quat_w = []
    for env_id in range(env.num_envs):
        object_1_quat_w.append(object_1.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][0], :4])
        object_2_quat_w.append(object_2.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][1], :4])
        object_3_quat_w.append(object_3.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][2], :4])
    object_1_quat_w = torch.stack(object_1_quat_w)
    object_2_quat_w = torch.stack(object_2_quat_w)
    object_3_quat_w = torch.stack(object_3_quat_w)

    return torch.cat((object_1_quat_w, object_2_quat_w, object_3_quat_w), dim=1)

# done
def object_obs(
    env: ManagerBasedRLEnv,
    object1_name: str,
    object2_name: str,
    object3_name: str,
    ee_frame_name: str,
):
    """
    Object observations (in world frame):
        object1 pos,
        object1 quat,
        object2 pos,
        object2 quat,
        object3 pos,
        object3 quat,
        gripper to object1,
        gripper to object2,
        gripper to object3,
        object1 to object2,
        object2 to object3,
        object1 to object3,
    """
    object1: RigidObject = env.scene[object1_name]
    object2: RigidObject = env.scene[object2_name]
    object3: RigidObject = env.scene[object3_name]
    ee_frame: FrameTransformer = env.scene[ee_frame_name]

    object_1_pos_w = object1.data.root_pos_w
    object_1_quat_w = object1.data.root_quat_w

    object_2_pos_w = object2.data.root_pos_w
    object_2_quat_w = object2.data.root_quat_w

    object_3_pos_w = object3.data.root_pos_w
    object_3_quat_w = object3.data.root_quat_w

    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    gripper_to_object_1 = object_1_pos_w - ee_pos_w
    gripper_to_object_2 = object_2_pos_w - ee_pos_w
    gripper_to_object_3 = object_3_pos_w - ee_pos_w

    object_1_to_2 = object_1_pos_w - object_2_pos_w
    object_2_to_3 = object_2_pos_w - object_3_pos_w
    object_1_to_3 = object_1_pos_w - object_3_pos_w

    return torch.cat(
        (
            object_1_pos_w - env.scene.env_origins,
            object_1_quat_w,
            object_2_pos_w - env.scene.env_origins,
            object_2_quat_w,
            object_3_pos_w - env.scene.env_origins,
            object_3_quat_w,
            gripper_to_object_1,
            gripper_to_object_2,
            gripper_to_object_3,
            object_1_to_2,
            object_2_to_3,
            object_1_to_3,
        ),
        dim=1,
    )

# done
def instance_randomize_object_obs(
    env: ManagerBasedRLEnv,
    object1_name: str,
    object2_name: str,
    object3_name: str,
    ee_frame_name: str,
):
    """
    Object observations (in world frame):
        object_1 pos,
        object_1 quat,
        object_2 pos,
        object_2 quat,
        object_3 pos,
        object_3 quat,
        gripper to object_1,
        gripper to object_2,
        gripper to object_3,
        object_1 to object_2,
        object_2 to object_3,
        object_1 to object_3,
    """
    if not hasattr(env, "rigid_objects_in_focus"):
        return torch.full((env.num_envs, 9), fill_value=-1)

    object_1: RigidObjectCollection = env.scene[object1_name]
    object_2: RigidObjectCollection = env.scene[object2_name]
    object_3: RigidObjectCollection = env.scene[object3_name]
    ee_frame: FrameTransformer = env.scene[ee_frame_name]

    object_1_pos_w = []
    object_2_pos_w = []
    object_3_pos_w = []
    object_1_quat_w = []
    object_2_quat_w = []
    object_3_quat_w = []
    for env_id in range(env.num_envs):
        object_1_pos_w.append(object_1.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][0], :3])
        object_2_pos_w.append(object_2.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][1], :3])
        object_3_pos_w.append(object_3.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][2], :3])
        object_1_quat_w.append(object_1.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][0], :4])
        object_2_quat_w.append(object_2.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][1], :4])
        object_3_quat_w.append(object_3.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][2], :4])
    object_1_pos_w = torch.stack(object_1_pos_w)
    object_2_pos_w = torch.stack(object_2_pos_w)
    object_3_pos_w = torch.stack(object_3_pos_w)
    object_1_quat_w = torch.stack(object_1_quat_w)
    object_2_quat_w = torch.stack(object_2_quat_w)
    object_3_quat_w = torch.stack(object_3_quat_w)

    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    gripper_to_object_1 = object_1_pos_w - ee_pos_w
    gripper_to_object_2 = object_2_pos_w - ee_pos_w
    gripper_to_object_3 = object_3_pos_w - ee_pos_w

    object_1_to_2 = object_1_pos_w - object_2_pos_w
    object_2_to_3 = object_2_pos_w - object_3_pos_w
    object_1_to_3 = object_1_pos_w - object_3_pos_w

    return torch.cat(
        (
            object_1_pos_w - env.scene.env_origins,
            object_1_quat_w,
            object_2_pos_w - env.scene.env_origins,
            object_2_quat_w,
            object_3_pos_w - env.scene.env_origins,
            object_3_quat_w,
            gripper_to_object_1,
            gripper_to_object_2,
            gripper_to_object_3,
            object_1_to_2,
            object_2_to_3,
            object_1_to_3,
        ),
        dim=1,
    )

# done
def ee_frame_pos(env: ManagerBasedRLEnv, ee_frame_name: str) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_name]
    ee_frame_pos = ee_frame.data.target_pos_w[:, 0, :] - env.scene.env_origins[:, 0:3]

    return ee_frame_pos

# done
def ee_frame_quat(env: ManagerBasedRLEnv, ee_frame_name: str) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_name]
    ee_frame_quat = ee_frame.data.target_quat_w[:, 0, :]

    return ee_frame_quat

# done
def gripper_pos(
    env: ManagerBasedRLEnv,
    robot_name: str = "robot"
) -> torch.Tensor:
    """
    Obtain the versatile gripper position of both Gripper and Suction Cup.
    """
    robot: Articulation = env.scene[robot_name]

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

# doing
def object_placed(
    env: ManagerBasedRLEnv,
    robot_name: str,
    box_name: str,
    object_name: str,
    xy_threshold: float = 0.04,
    height_threshold: float = 0.05,
) -> torch.Tensor:
    """ Check if an object is placed into the box."""
    robot: Articulation = env.scene[robot_name]
    box: RigidObject = env.scene[box_name]
    object: RigidObject = env.scene[object_name]

    obj_pos = object.data.root_pos_w
    box_pos = box.data.root_pos_w
    box_dims = box.data.root_dims_w

    

# done
def object_grasped(
    env: ManagerBasedRLEnv,
    robot_name: str,
    ee_frame_name: str,
    object_name: str,
    diff_threshold: float = 0.06,
) -> torch.Tensor:
    """Check if an object is grasped by the specified robot."""

    robot: Articulation = env.scene[robot_name]
    ee_frame: FrameTransformer = env.scene[ee_frame_name]
    object: RigidObject = env.scene[object_name]

    object_pos = object.data.root_pos_w
    end_effector_pos = ee_frame.data.target_pos_w[:, 0, :]
    pose_diff = torch.linalg.vector_norm(object_pos - end_effector_pos, dim=1)

    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        surface_gripper = env.scene.surface_grippers["surface_gripper"]
        suction_cup_status = surface_gripper.state.view(-1, 1)  # 1: closed, 0: closing, -1: open
        suction_cup_is_closed = (suction_cup_status == 1).to(torch.float32)
        grasped = torch.logical_and(suction_cup_is_closed, pose_diff < diff_threshold)

    else:
        if hasattr(env.cfg, "gripper_joint_names"):
            gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
            assert len(gripper_joint_ids) == 2, "Observations only support parallel gripper for now"

            grasped = torch.logical_and(
                pose_diff < diff_threshold,
                torch.abs(
                    robot.data.joint_pos[:, gripper_joint_ids[0]]
                    - torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device)
                )
                > env.cfg.gripper_threshold,
            )
            grasped = torch.logical_and(
                grasped,
                torch.abs(
                    robot.data.joint_pos[:, gripper_joint_ids[1]]
                    - torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device)
                )
                > env.cfg.gripper_threshold,
            )

    return grasped

# done
def object_poses_in_base_frame(
    env: ManagerBasedRLEnv,
    object1_name: str,
    object2_name: str,
    object3_name: str,
    robot_name: str = "robot",
    return_key: Literal["pos", "quat", None] = None,
) -> torch.Tensor:
    """The position and orientation of the objects in the robot base frame."""

    object_1: RigidObject = env.scene[object1_name]
    object_2: RigidObject = env.scene[object2_name]
    object_3: RigidObject = env.scene[object3_name]

    pos_object_1_world = object_1.data.root_pos_w
    pos_object_2_world = object_2.data.root_pos_w
    pos_object_3_world = object_3.data.root_pos_w

    quat_object_1_world = object_1.data.root_quat_w
    quat_object_2_world = object_2.data.root_quat_w
    quat_object_3_world = object_3.data.root_quat_w

    robot: Articulation = env.scene[robot_name]
    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w

    pos_object_1_base, quat_object_1_base = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, pos_object_1_world, quat_object_1_world
    )
    pos_object_2_base, quat_object_2_base = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, pos_object_2_world, quat_object_2_world
    )
    pos_object_3_base, quat_object_3_base = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, pos_object_3_world, quat_object_3_world
    )

    pos_objects_base = torch.cat((pos_object_1_base, pos_object_2_base, pos_object_3_base), dim=1)
    quat_objects_base = torch.cat((quat_object_1_base, quat_object_2_base, quat_object_3_base), dim=1)

    if return_key == "pos":
        return pos_objects_base
    elif return_key == "quat":
        return quat_objects_base
    else:
        return torch.cat((pos_objects_base, quat_objects_base), dim=1)

def object_abs_obs_in_base_frame(
    env: ManagerBasedRLEnv,
    object1_name: str,
    object2_name: str,
    object3_name: str,
    ee_frame_name: str,
    robot_name: str = "robot",
):
    """
    Object Abs observations (in base frame): remove the relative observations,
    and add abs gripper pos and quat in robot base frame
        object_1 pos,
        object_1 quat,
        object_2 pos,
        object_2 quat,
        object_3 pos,
        object_3 quat,
        gripper pos,
        gripper quat,
    """
    object_1: RigidObject = env.scene[object1_name]
    object_2: RigidObject = env.scene[object2_name]
    object_3: RigidObject = env.scene[object3_name]
    ee_frame: FrameTransformer = env.scene[ee_frame_name]
    robot: Articulation = env.scene[robot_name]

    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w

    object_1_pos_w = object_1.data.root_pos_w
    object_1_quat_w = object_1.data.root_quat_w

    object_2_pos_w = object_2.data.root_pos_w
    object_2_quat_w = object_2.data.root_quat_w

    object_3_pos_w = object_3.data.root_pos_w
    object_3_quat_w = object_3.data.root_quat_w

    pos_object_1_base, quat_object_1_base = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, object_1_pos_w, object_1_quat_w
    )
    pos_object_2_base, quat_object_2_base = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, object_2_pos_w, object_2_quat_w
    )
    pos_object_3_base, quat_object_3_base = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, object_3_pos_w, object_3_quat_w
    )

    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    ee_quat_w = ee_frame.data.target_quat_w[:, 0, :]
    ee_pos_base, ee_quat_base = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)

    return torch.cat(
        (
            pos_object_1_base,
            quat_object_1_base,
            pos_object_2_base,
            quat_object_2_base,
            pos_object_3_base,
            quat_object_3_base,
            ee_pos_base,
            ee_quat_base,
        ),
        dim=1,
    )

def ee_frame_pose_in_base_frame(
    env: ManagerBasedRLEnv,
    ee_frame_name:str,
    robot_name: str = "robot",
    return_key: Literal["pos", "quat", None] = None,
) -> torch.Tensor:
    """
    The end effector pose in the robot base frame.
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_name]
    ee_frame_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    ee_frame_quat_w = ee_frame.data.target_quat_w[:, 0, :]

    robot: Articulation = env.scene[robot_name]
    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w
    ee_pos_in_base, ee_quat_in_base = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, ee_frame_pos_w, ee_frame_quat_w
    )

    if return_key == "pos":
        return ee_pos_in_base
    elif return_key == "quat":
        return ee_quat_in_base
    else:
        return torch.cat((ee_pos_in_base, ee_quat_in_base), dim=1)
