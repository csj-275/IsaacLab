# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.devices.spacemouse import Se3SpaceMouseCfg
from isaaclab.devices.xrobotoolkit import XRoboToolkitDeviceCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.stack import mdp
from isaaclab_tasks.manager_based.manipulation.stack.config.franka import stack_joint_pos_env_cfg
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events
from isaaclab_tasks.manager_based.piper_grab.mdp.mimic_joint_actions import MimicBinaryJointPositionActionCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.piper import PIPER_STANDARD_WITH_GRIPPER_HIGH_PD_CFG  # isort: skip


PIPER_D435_COLOR_INTRINSIC_640X480 = [
    605.519378662109,
    0.0,
    320.0,
    0.0,
    605.519378662109,
    240.0,
    0.0,
    0.0,
    1.0,
]
"""Isaac Sim compatible D435 color intrinsics at 640x480 from ``rs-enumerate-devices -c``.

Omniverse ignores camera aperture offsets and non-square pixels, so this uses the
measured Color 640x480 average focal length with a centered principal point.
"""

PIPER_TABLE_CAM_DEFAULT_POS = (0.4804880058635685,-0.6895463223929142,1.2665301203868786)
"""Default manually tuned table camera position in environment-local coordinates, in meters."""

PIPER_TABLE_CAM_DEFAULT_ROT_OPENGL =(0.9724436366453746, 0.2277544386785688, 0.04605009910256881, 0.018991513440805734)
"""Default table camera orientation as an OpenGL/USD quaternion ``(w, x, y, z)``.

Derived from the manually tuned viewport XYZ Euler orientation
``(1.27572, 0.081, 0.169)`` in radians.
"""


@configclass
class EventCfg:
    """Configuration for Piper stack reset events."""

    init_piper_arm_pose = EventTerm(
        func=franka_stack_events.set_default_joint_pose,
        mode="reset",
        params={"default_pose": [0.0, 1.0, -0.6, 0.0, 1.35, 0.0, 0.05, -0.05]},
    )

    randomize_piper_joint_state = EventTerm(
        func=franka_stack_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.02,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    randomize_cube_positions = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {"x": (0.15, 0.35), "y": (-0.10, 0.10), "z": (0.0203, 0.0203), "yaw": (-1.0, 1.0, 0)},
            "min_separation": 0.1,
            "asset_cfgs": [SceneEntityCfg("cube_1"), SceneEntityCfg("cube_2"), SceneEntityCfg("cube_3")],
        },
    )


@configclass
class PiperCubeStackEnvCfg(stack_joint_pos_env_cfg.FrankaCubeStackEnvCfg):
    """Configuration for the Piper cube stack environment using relative differential IK."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.decimation = 2
        self.sim.render_interval = 2


        # Set Piper as robot
        self.events = EventCfg()
        self.scene.robot = PIPER_STANDARD_WITH_GRIPPER_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]

        # Set actions for Piper. The gripper is asymmetric: joint7 opens positive, joint8 opens negative.
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["joint[1-6]"],
            body_name="link6",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.13503]),
        )
        self.actions.gripper_action = MimicBinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint7"],
            open_command_expr={"joint7": 0.05},
            close_command_expr={"joint7": -0.05},
            mimic_joint_names=["joint8"],
            mimic_multiplier=-1.0,
            max_speed_per_step=0.005,  # 调这里：越大越快
        )
        self.gripper_joint_names = ["joint[7-8]"]
        self.gripper_open_val = 0.05
        self.gripper_open_vals = [0.05, -0.05]
        self.gripper_threshold = 0.005

        # Listen to the end-effector frame used by differential IK and stack observations.
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/arm_base",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/link6",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.13503]),
                ),
            ],
        )

        # Set wrist camera attached to the Piper URDF camera link.
        self.scene.wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/camera_link/wrist_cam",
            update_period=0.0,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
                intrinsic_matrix=PIPER_D435_COLOR_INTRINSIC_640X480,
                width=640,
                height=480,
                clipping_range=(0.1, 2.0),
                focus_distance=400.0,
            ),
            # Convert the URDF RealSense camera_link frame to the optical frame expected by IsaacLab cameras.
            offset=CameraCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.0),
                rot=(0.5, -0.5, 0.5, -0.5),
                convention="ros",
            ),
        )
        # Set table view camera to the manually tuned default view. The teleop script can optionally
        # override this at runtime with the frustum-based placement helper.
        self.scene.table_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/table_cam",
            update_period=0.0,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
                intrinsic_matrix=PIPER_D435_COLOR_INTRINSIC_640X480,
                width=640,
                height=480,
                clipping_range=(0.05, 10.0),
                focus_distance=400.0,
            ),
            offset=CameraCfg.OffsetCfg(
                pos=PIPER_TABLE_CAM_DEFAULT_POS,
                rot=PIPER_TABLE_CAM_DEFAULT_ROT_OPENGL,
                convention="opengl",
            ),
        )
        self.num_rerenders_on_reset = 3
        self.sim.render.antialiasing_mode = "DLAA"

        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.03,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
                "spacemouse": Se3SpaceMouseCfg(
                    pos_sensitivity=0.05,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
                "xrobotoolkit": XRoboToolkitDeviceCfg(
                    control_mode="absolute",
                    mapping_mode="world_frame_calibrated",
                    pos_sensitivity=1.0,
                    rot_sensitivity=1.0,
                    sim_device=self.sim.device,
                ),
            }
        )
