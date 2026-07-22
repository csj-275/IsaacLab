# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
# xrobotool
from isaaclab.devices.xrobotoolkit import XRoboToolkitDeviceCfg

from isaaclab.assets import RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices import DeviceBase, DevicesCfg, OpenXRDeviceCfg, Se3KeyboardCfg
from isaaclab.devices.openxr.retargeters.manipulator.gripper_retargeter import GripperRetargeterCfg
from isaaclab.devices.openxr.retargeters.manipulator.se3_rel_retargeter import Se3RelRetargeterCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.spawners.materials.visual_materials_cfg import PreviewSurfaceCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.piper_grab import mdp
from isaaclab_tasks.manager_based.piper_grab.mdp import piper_grab_events
from isaaclab_tasks.manager_based.piper_grab.V1 import mdp as V1_mdp

from .. import grab_joint_pos_env_cfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.piper import PIPER_STANDARD_WITH_GRIPPER_HIGH_PD_CFG  # isort: skip
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

# _MUG_USD_PATH = f"{ISAACLAB_NUCLEUS_DIR}/Objects/Mug/mug.usd"

_MUG_USD_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../../../../../usd/bottle/bottle.usd")
)

@configclass
class SubtaskCfg(ObsGroup):
    """Observations for subtask group shared across V1 variants."""

    grasp_1 = ObsTerm(
        func=mdp.object_grasped,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "object_cfg": SceneEntityCfg("object_1"),
            "diff_threshold": 0.05,
        },
    )
    placed_1 = ObsTerm(
        func=mdp.object_a_is_into_b,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "object_a_cfg": SceneEntityCfg("object_1"),
            "object_b_cfg": SceneEntityCfg("box"),
            "xy_threshold": 0.05,
            "height_diff": 0.0,
            "height_threshold": 0.05,
            "gripper_threshold": 0.03,  # 独立阈值，比 grasp 宽松
        },
    )
    grasp_2 = ObsTerm(
        func=mdp.object_grasped,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "object_cfg": SceneEntityCfg("mug"),
            "diff_threshold": 0.05,
        },
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class ObservationsCfg:
    """Observation specifications for V1 two-stage task (IK-rel, state-only)."""

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
        mug_positions = ObsTerm(
            func=mdp.object_poses_in_base_frame, params={"object_cfg": SceneEntityCfg("mug"), "return_key": "pos"}
        )
        mug_orientations = ObsTerm(
            func=mdp.object_poses_in_base_frame,
            params={"object_cfg": SceneEntityCfg("mug"), "return_key": "quat"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class TerminationsCfg:
    """Termination conditions for V1 task."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_1_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object_1")},
    )
    mug_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("mug")},
    )
    box_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("box")},
    )
    success = DoneTerm(
        func=V1_mdp.objects_a_and_b_are_into_c,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "object_a_cfg": SceneEntityCfg("object_1"),
            "object_b_cfg": SceneEntityCfg("mug"),
            "object_c_cfg": SceneEntityCfg("box"),
        },
    )


@configclass
class PiperGrabEnvCfg(grab_joint_pos_env_cfg.PiperGrabEnvCfg):
    """Configuration for V1 two-stage pick-and-place IK-rel environment."""

    observations: ObservationsCfg = ObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Override robot to high-PD variant for IK tracking
        self.scene.robot = PIPER_STANDARD_WITH_GRIPPER_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]

        # IK-rel action
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["joint[1-6]"],
            body_name="link6",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )

        # Gripper config
        self.gripper_joint_names = ["joint7", "joint8"]
        
        self.gripper_open_vals = [0.05, -0.05]
        self.gripper_threshold = 0.015  # 放宽阈值，replay 时夹爪速率限制导致实际位置无法精确到达 open_vals
        # Add mug to scene
        mug_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )

        self.scene.mug = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/mug",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, -0.15, 0.203), rot=(1, 0, 0, 0)),
            spawn=UsdFileCfg(
                usd_path=_MUG_USD_PATH,
                # scale=(0.8, 0.8, 0.8),
                scale=(1, 1, 1),
                rigid_props=mug_properties,
                semantic_tags=[("class", "mug")],
            ),
        )

        # Combined cube + mug + box randomization with per-object pose ranges to avoid overlaps
        del self.events.randomize_cube_positions
        del self.events.randomize_box_positions

        self.events.randomize_cube_and_mug_and_box_poses = EventTerm(
            func=piper_grab_events.randomize_object_pose,
            mode="reset",
            params={
                "pose_ranges": [
                    # {"x": (0.2, 0.35), "y": (-0.05, 0.15), "z": (0.0203, 0.0203), "yaw": (-0.785, 0.785)},  # cube
                    # {"x": (0.2, 0.35), "y": (-0.05, 0.15), "z": (0.0000, 0.0000), "yaw": (-0.785, 0.785)},  # mug
                    # {"x": (0.1, 0.3), "y": (0.05, 0.35), "z": (0.0000, 0.0000), "yaw": (-0.785, 0.785)},  # box
                    {"x": (0.3, 0.35), "y": (0.1, 0.15), "z": (0.0203, 0.0203), "yaw": (-0.785, 0.785)},  # cube
                    {"x": (0.2, 0.25), "y": (0.1, 0.15), "z": (0.0000, 0.0000), "yaw": (-0.785, 0.785)},  # mug
                    {"x": (0.15, 0.2), "y": (0.25, 0.3), "z": (0.0000, 0.0000), "yaw": (-0.785, 0.785)},  # box
                ],
                "min_separation": 0.12,
                "asset_cfgs": [SceneEntityCfg("object_1"), SceneEntityCfg("mug"), SceneEntityCfg("box")],
            },
        )

        # Object visual materials
        self.scene.mug.spawn.visual_material = PreviewSurfaceCfg(
            diffuse_color=(0.85, 0.45, 0.10),  # orange
            roughness=0.4,
            metallic=0.0,
        )
        self.scene.box.spawn.visual_material = PreviewSurfaceCfg(
            diffuse_color=(0.55, 0.35, 0.15),  # brown/cardboard
            roughness=0.5,
            metallic=0.0,
        )

        self.sim.dt = 1 / 150
        self.decimation = 5
        self.sim.render_interval = 3
        # Teleop devices
        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.03,
                    rot_sensitivity=0.15,
                    sim_device=self.sim.device,
                ),
            }
        )
