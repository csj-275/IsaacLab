# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for AgileX Piper robots.

The following configurations are available:

* :obj:`PIPER_STANDARD_WITH_GRIPPER_CFG`: Piper standard arm with the two-finger gripper.
* :obj:`PIPER_STANDARD_WITH_GRIPPER_HIGH_PD_CFG`: Piper standard arm with stiffer PD control for IK.
"""

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##

PIPER_ASSET_ROOT = Path(__file__).resolve().parents[1] / "data" / "piper"
"""Repository-local root containing Piper URDF packages and meshes."""

PIPER_STANDARD_WITH_GRIPPER_URDF = str(
    PIPER_ASSET_ROOT / "piper_description" / "urdf" / "piper_description_v100_realsense_camera_v2.urdf"
)
"""URDF path for the standard Piper variant with gripper and RealSense camera links."""

PIPER_USD_CONVERSION_DIR = "/tmp/isaaclab_piper_assets"
"""Output directory for the generated Piper USD."""


def _prepend_ros_package_path(path: Path) -> None:
    """Expose repository-local URDF packages for package:// mesh resolution."""
    path_text = str(path)
    current = os.environ.get("ROS_PACKAGE_PATH", "")
    entries = [entry for entry in current.split(os.pathsep) if entry]
    if path_text not in entries:
        os.environ["ROS_PACKAGE_PATH"] = os.pathsep.join([path_text, *entries])


_prepend_ros_package_path(PIPER_ASSET_ROOT)


PIPER_STANDARD_WITH_GRIPPER_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=PIPER_STANDARD_WITH_GRIPPER_URDF,
        usd_dir=PIPER_USD_CONVERSION_DIR,
        usd_file_name="piper_description_v100_realsense_camera_v2.usd",
        fix_base=True,
        root_link_name="arm_base",
        merge_fixed_joints=False,
        make_instanceable=False,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint1": 0.0,
            "joint2": 1.0,
            "joint3": -1.2,
            "joint4": 0.0,
            "joint5": 0.8,
            "joint6": 0.0,
            "joint7": 0.05,
            "joint8": -0.05,
        },
    ),
    actuators={
        "piper_arm": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-6]"],
            effort_limit_sim=100.0,
            velocity_limit_sim=5.0,
            stiffness=80.0,
            damping=4.0,
        ),
        "piper_gripper": ImplicitActuatorCfg(
            joint_names_expr=["joint[7-8]"],
            effort_limit_sim=10.0,
            velocity_limit_sim=1.0,
            stiffness=2.0e3,
            damping=1.0e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of the standard Piper robot with gripper."""


PIPER_STANDARD_WITH_GRIPPER_HIGH_PD_CFG = PIPER_STANDARD_WITH_GRIPPER_CFG.copy()
PIPER_STANDARD_WITH_GRIPPER_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
PIPER_STANDARD_WITH_GRIPPER_HIGH_PD_CFG.actuators["piper_arm"].stiffness = 400.0
PIPER_STANDARD_WITH_GRIPPER_HIGH_PD_CFG.actuators["piper_arm"].damping = 80.0
"""Configuration of the standard Piper robot with stiffer PD control for differential IK."""
