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

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##

PIPER_ISAAC_SIM_ROOT = os.environ.get("PIPER_ISAAC_SIM_ROOT", "/workspace/piper_isaac_sim")
"""Root directory of the mounted piper_isaac_sim checkout."""

PIPER_STANDARD_WITH_GRIPPER_USD = os.path.join(PIPER_ISAAC_SIM_ROOT, "USD", "piper_v2.usd")
"""USD path for the standard Piper variant with gripper."""

PIPER_STANDARD_WITH_GRIPPER_WRAPPED_USD = os.environ.get(
    "PIPER_STANDARD_WITH_GRIPPER_WRAPPED_USD",
    os.path.join(os.environ.get("ISAACLAB_PIPER_WRAPPER_DIR", "/tmp/isaaclab_piper_assets"), "piper.usda"),
)
"""Thin wrapper USD path that exposes Piper's /piper_camera prim as defaultPrim."""


def _ensure_piper_wrapper(source_usd: str, wrapper_usd: str):
    """Create a defaultPrim wrapper for the upstream Piper USD, which has no defaultPrim."""
    os.makedirs(os.path.dirname(wrapper_usd), exist_ok=True)
    source_usd = os.path.abspath(source_usd)
    with open(wrapper_usd, "w", encoding="utf-8") as f:
        f.write(
            '#usda 1.0\n'
            '(\n'
            '    defaultPrim = "piper"\n'
            '    metersPerUnit = 1\n'
            '    upAxis = "Z"\n'
            ')\n'
            '\n'
            'def Xform "piper" (\n'
            f'    references = @{source_usd}@</piper_camera>\n'
            ')\n'
            '{\n'
            '}\n'
        )


_ensure_piper_wrapper(PIPER_STANDARD_WITH_GRIPPER_USD, PIPER_STANDARD_WITH_GRIPPER_WRAPPED_USD)


PIPER_STANDARD_WITH_GRIPPER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=PIPER_STANDARD_WITH_GRIPPER_WRAPPED_USD,
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
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint1": 0.0,
            "joint2": 1.0,
            "joint3": -1.2,
            "joint4": 0.0,
            "joint5": 0.8,
            "joint6": 0.0,
            "joint7": 0.035,
            "joint8": -0.035,
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
