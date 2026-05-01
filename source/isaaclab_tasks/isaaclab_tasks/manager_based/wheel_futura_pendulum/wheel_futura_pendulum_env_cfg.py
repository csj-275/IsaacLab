# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
import math
from . import mdp

##
# Pre-defined configs
##

from isaaclab_assets import WHEEL_FUTURA_PENDULUM_CFG  # isort:skip

##
# Scene definition
##

@configclass
class WheelFuturaPendulumSceneCfg(InteractiveSceneCfg):
    """Configuration for a wheel-futura-pendulum scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # articulation
    robot: ArticulationCfg = WHEEL_FUTURA_PENDULUM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot", 
        joint_names=["wheel_joint"], 
        scale=10.0)

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        # target pos
        # target_pos = ObsTerm(func=mdp.return_target_joint_pos)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    reset_joint1_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint1"]),
            "position_range": (-1.0*math.pi, 1.0*math.pi),
            "velocity_range": (-0.5*math.pi, 0.5*math.pi),
        },
    )

    reset_joint2_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint2"]),
            "position_range": (-1.0*math.pi, 1.0*math.pi),
            "velocity_range": (-0.5*math.pi, 0.5*math.pi),
        },
    )

    reset_wheel_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["wheel_joint"]),
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.5, 0.5),
        },
    )

@configclass
class RewardsCfg:
    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    # primary task：minimize the error
    tracking_error = RewTerm(
        func=mdp.joint_pos_target_l2,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint1"])},
    )

    tracking_error = RewTerm(
        func=mdp.joint_pos_target_l2,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint2"]), "target": -math.pi},
    )

    # 速度惩罚，抑制残余振荡
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint[1-2]"])},
    )

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["wheel_joint"])},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


##
# Environment configuration
##

@configclass
class WheelFuturaPendulumEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: WheelFuturaPendulumSceneCfg = WheelFuturaPendulumSceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation