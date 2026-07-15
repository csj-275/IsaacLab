# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices.device_base import DeviceBase, DevicesCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.devices.openxr.openxr_device import OpenXRDeviceCfg
from isaaclab.devices.openxr.retargeters.manipulator.gripper_retargeter import GripperRetargeterCfg
from isaaclab.devices.openxr.retargeters.manipulator.se3_rel_retargeter import Se3RelRetargeterCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import mdp
from . import grab_joint_pos_env_cfg
##
# Pre-defined configs
##
from isaaclab_assets.robots.piper import PIPER_HIGH_PD_CFG  # isort: skip


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object = ObsTerm(func=mdp.object_obs)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)
        object_1_positions = ObsTerm(
            func=mdp.object_poses_in_base_frame,
            params={"object_cfg": SceneEntityCfg("object_1"), "return_key": "pos"},
        )
        object_1_orientations = ObsTerm(
            func=mdp.object_poses_in_base_frame,
            params={"object_cfg": SceneEntityCfg("object_1"), "return_key": "quat"},
        )
        box_positions = ObsTerm(
            func=mdp.object_poses_in_base_frame, params={"object_cfg": SceneEntityCfg("box"), "return_key": "pos"}
        )
        box_orientations = ObsTerm(
            func=mdp.object_poses_in_base_frame,
            params={"object_cfg": SceneEntityCfg("box"), "return_key": "quat"},
        )
        

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class RGBCameraPolicyCfg(ObsGroup):
        """Observations for policy group with RGB images."""

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""

        grasp_1 = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("object_1"),
            },
        )
        placed_1 = ObsTerm(
            func=mdp.object_a_is_into_b,
            params={
                    "robot_cfg": SceneEntityCfg("robot"),
                    "object_a_cfg": SceneEntityCfg("object_1"),
                    "object_b_cfg": SceneEntityCfg("box"),
                    "xy_threshold": 0.08,
                    "height_diff": 0.0,
                    "height_threshold": 0.08,
                },
        )
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    rgb_camera: RGBCameraPolicyCfg = RGBCameraPolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class PiperGrabSkillgenEnvCfg(grab_joint_pos_env_cfg.PiperGrabEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Override observations with SkillGen-specific config
        self.observations = ObservationsCfg()

        # Set Piper as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        self.scene.robot = PIPER_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # [IK action]
        # Set actions for the specific robot type (piper)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["joint[1-6]"],
            body_name="link6",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        ) # 0.135/0.107 未测试


        self.gripper_joint_names = ["joint7", "joint8"]
        self.gripper_open_vals = [0.05, -0.05]
        self.gripper_threshold = 0.01

        self.teleop_devices = DevicesCfg(
            devices={
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        Se3RelRetargeterCfg(
                            bound_hand=DeviceBase.TrackingTarget.HAND_RIGHT,
                            zero_out_xy_rotation=True,
                            use_wrist_rotation=False,
                            use_wrist_position=True,
                            delta_pos_scale_factor=10.0,
                            delta_rot_scale_factor=10.0,
                            sim_device=self.sim.device,
                        ),
                        GripperRetargeterCfg(
                            bound_hand=DeviceBase.TrackingTarget.HAND_RIGHT, sim_device=self.sim.device
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.05,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
            }
        )

        # Apply skillgen-specific cube position randomization
        self.events.randomize_cube_positions.params["pose_range"] = {
            "x": (0.45, 0.6),
            "y": (-0.23, 0.23),
            "z": (0.0203, 0.0203),
            "yaw": (-1.0, 1, 0),
        }

        # Set the offset for the end effector to be 0.0
        for f in self.scene.ee_frame.target_frames:
            if f.name == "end_effector":
                f.offset.pos = (0.0, 0.0, 0.0)
                break
