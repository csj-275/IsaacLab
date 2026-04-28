# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to create a simple environment with a cartpole. It combines the concepts of
scene, action, observation and event managers to create an environment.

.. code-block:: bash

    ./isaaclab.sh -p scripts/tutorials/03_envs/create_piper_base_env.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on creating a cartpole base environment.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math

import torch

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.piper.piper_env_cfg import PiperSceneCfg

@configclass
class ActionsCfg:
    """Action specifications for the environment."""

    arm_actions = mdp.JointPositionActionCfg(
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
class PiperEnvCfg(ManagerBasedEnvCfg):
    """Configuration for the cartpole environment."""

    # Scene settings
    scene = PiperSceneCfg(num_envs=1024, env_spacing=2.5)
    # Basic settings
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # viewer settings
        self.viewer.eye = (4.5, 0.0, 6.0)
        self.viewer.lookat = (0.0, 0.0, 2.0)
        # step settings
        self.decimation = 4  # env step every 4 sim steps: 200Hz / 4 = 50Hz
        # simulation settings
        self.sim.dt = 0.005  # sim step every 5ms: 200Hz


def main():
    """Main function."""
    # parse the arguments
    env_cfg = PiperEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    # setup base environment
    env = ManagerBasedEnv(cfg=env_cfg)
    robot = env.scene["robot"]

    # simulate physics
    count = 0
    gripper_state = 1  # 1 = open, -1 = close
    device = env.device
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 300 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
                gripper_state = -gripper_state
                close_str = 'CLOSE' if gripper_state < 0 else 'OPEN'
                print(f"[INFO]: Gripper command value = {gripper_state} ({close_str})")
            # sample random actions
            arm_actions = torch.randn(env.num_envs, 6, device=device)

            gripper_action = 10*torch.full((env.num_envs, 1), gripper_state, device=device)
            joint_actions = torch.cat([arm_actions, gripper_action], dim=1)

            # step the environment
            obs, _ = env.step(joint_actions)

            current_joint_pos = robot.data.joint_pos
            gripper_pos = current_joint_pos[:, -2:] 
            if env.num_envs == 1:
                pos_str = f"J7: {gripper_pos[0, 0]:.4f}, J8: {gripper_pos[0, 1]:.4f}"
            else:
                mean_pos = gripper_pos.mean(dim=0)
                pos_str = f"Mean - J7: {mean_pos[0]:.4f}, J8: {mean_pos[1]:.4f}"
            
            if count % 100 == 0:
                current_state = 'CLOSED' if gripper_state < 0 else 'OPENED'
                print(f"[INFO]: Gripper State: {current_state} | "
                      f"Actual Pos: {pos_str}")
            # update counter
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
