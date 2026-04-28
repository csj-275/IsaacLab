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
from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_tasks.manager_based.piper.piper_ik_env_cfg import PiperEnvCfg


def main():
    """Main function."""
    # parse the arguments
    env_cfg = PiperEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    # setup base environment
    env = ManagerBasedRLEnv(cfg=env_cfg)
    robot = env.scene["robot"]

    # simulate physics
    count = 0
    # gripper_state 逻辑保持不变
    gripper_state = 1  # 1 = open, -1 = close
    device = env.device
    
    while simulation_app.DifferentialInverseKinematicsActionCfgis_running():
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
            
            # IK relative_mode-6(True),7(False)
            arm_actions = torch.zeros(env.num_envs, 6, device=device) 
            arm_actions[:, 2] = 0.1
            gripper_action = 10 * torch.full((env.num_envs, 1), gripper_state, device=device)

            joint_actions = torch.cat([arm_actions, gripper_action], dim=1)

            # step the environment
            obs = env.step(joint_actions)

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
                      f"Actual Pos: {pos_str} | "
                      f"Action sent: {arm_actions[0].cpu().numpy()}")
            
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
