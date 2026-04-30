# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
    Author: csj
    Created: 26-04-27
    Modified: 26-04-27
    Description: Configuration the env for the piper robot.
"""


import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
# from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg

# cube 
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg


from . import mdp

##
# Pre-defined configs
##

from isaaclab_assets.robots.piper import PIPER_CFG  # isort:skip

##
# Scene definition
##

@configclass
class PiperSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=GroundPlaneCfg(size=(100.0, 100.0)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, -1.05)),
    )

    # robot
    robot: ArticulationCfg = PIPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0, 0), rot=(0.707, 0, 0, 0.707)),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )



    
    
@configclass
class ActionsCfg:
    """Action specifications for the environment."""

    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=["joint[1-6]"],
        scale=1.0,
    )

    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint[7-8]"],        
        open_command_expr={
            "joint7": 0.1,
            "joint8": -0.1,
        },
        close_command_expr={
            "joint7": -0.1,
            "joint8": 0.1,
        },
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # on reset: reset arm joints to initial positions
    reset_arm_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint[1-6]"]),
            "position_range": (0.0, 0.0),  # reset to zero offset
            "velocity_range": (0.0, 0.0),
        },
    )

    # on reset: reset gripper joints to initial positions
    reset_gripper_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint[7-8]"]),
            "position_range": (0.0, 0.0),  # reset to zero offset
            "velocity_range": (0.0, 0.0),
        },
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    alive = RewTerm(func=mdp.is_alive, weight=1.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

##
# Environment configuration
##

@configclass
class PiperEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the piper environment."""

    # Scene settings
    scene: PiperSceneCfg = PiperSceneCfg(num_envs=1024, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()

    # MDP settings
    # rewards: RewardsCfg = RewardsCfg()
    # terminations: TerminationsCfg = TerminationsCfg()
    
    rewards = None
    terminations = None

    def __post_init__(self):
        """Post initialization."""
        # viewer settings
        self.viewer.eye = (4.5, 0.0, 6.0)
        self.viewer.lookat = (0.0, 0.0, 2.0)
        # step settings
        self.decimation = 4  # env step every 4 sim steps: 200Hz / 4 = 50Hz
        # simulation settings
        self.sim.dt = 0.005  # sim step every 5ms: 200Hz
        self.episode_length_s = 20.0 
