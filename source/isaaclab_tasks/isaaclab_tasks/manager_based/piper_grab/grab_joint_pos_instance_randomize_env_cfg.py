# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


from isaaclab_tasks.manager_based.piper_grab import mdp
from isaaclab_tasks.manager_based.piper_grab.mdp import piper_grab_events
from isaaclab_tasks.manager_based.piper_grab.grab_instance_randomize_env_cfg import GrabInstanceRandomizeEnvCfg


##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.piper import PIPER_CFG  # isort: skip


@configclass
class EventCfg:
    """Configuration for events."""

    init_franka_arm_pose = EventTerm(
        func=piper_grab_events.set_default_joint_pose,
        mode="startup",
        params={
            "default_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
    )

    randomize_franka_joint_state = EventTerm(
        func=piper_grab_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.02,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    randomize_cubes_in_focus = EventTerm(
        func=piper_grab_events.randomize_rigid_objects_in_focus,
        mode="reset",
        params={
            "asset_cfgs": [SceneEntityCfg("cube_1")],
            "out_focus_state": torch.tensor([10.0, 10.0, 10.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "pose_range": {"x": (0.4, 0.6), "y": (-0.10, 0.10), "z": (0.0203, 0.0203), "yaw": (-1.0, 1, 0)},
            "min_separation": 0.1,
        },
    )


@configclass
class PiperGrabInstanceRandomizeEnvCfg(GrabInstanceRandomizeEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set events
        self.events = EventCfg()

        # Set Piper as robot
        self.scene.robot = PIPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Reduce the number of environments due to camera resources
        self.scene.num_envs = 2

        # Set actions for the specific robot type (piper)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["joint[1-6]"],scale=1.0
        )

        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint[7-8]"],        
            open_command_expr={"joint7": 0.1,"joint8": -0.1,},
            close_command_expr={"joint7": -0.1,"joint8": 0.1},
        )
        # utilities for gripper status check
        self.gripper_joint_names = ["joint[7-8]"]
        self.gripper_open_val = 0.04 # need to check
        self.gripper_threshold = 0.005

        # Rigid body properties of each cube
        cube_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )

        # Set each stacking cube to be a collection of rigid objects
        cube_1_config_dict = {
            "blue_cube": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cube_1_Blue",
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 0.0203), rot=(1, 0, 0, 0)),
                spawn=UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd",
                    scale=(1.0, 1.0, 1.0),
                    rigid_props=cube_properties,
                ),
            ),
            "red_cube": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cube_1_Red",
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 0.0403), rot=(1, 0, 0, 0)),
                spawn=UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd",
                    scale=(1.0, 1.0, 1.0),
                    rigid_props=cube_properties,
                ),
            ),
        }


        self.scene.object_1 = RigidObjectCollectionCfg(rigid_objects=cube_1_config_dict)

        # Listens to the required transforms
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
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.0),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/link7",
                    name="tool_leftfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.135),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/link8",
                    name="tool_rightfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.135),
                    ),
                ),
            ],
        )
