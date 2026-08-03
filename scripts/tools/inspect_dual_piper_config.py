# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Launch dual-arm Piper environment with GUI for visual inspection.

.. code-block:: bash

    ./isaaclab.sh -p scripts/tools/inspect_dual_piper_config.py
    ./isaaclab.sh -p scripts/tools/inspect_dual_piper_config.py --env ik_rel
    ./isaaclab.sh -p scripts/tools/inspect_dual_piper_config.py --env visuomotor --enable_cameras

"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Inspect dual-arm Piper environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--env", type=str, default="joint_pos",
                    choices=["joint_pos", "ik_rel", "visuomotor"],
                    help="Which environment config to use.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from isaaclab.envs import ManagerBasedRLEnv


def main():
    if args_cli.env == "joint_pos":
        from isaaclab_tasks.manager_based.dual_piper.pick_place.dual_piper_grab_joint_pos_env_cfg import (
            DualPiperGrabJointPosEnvCfg,
        )
        env_cfg = DualPiperGrabJointPosEnvCfg()
        use_ik = False
    elif args_cli.env == "ik_rel":
        from isaaclab_tasks.manager_based.dual_piper.pick_place.dual_piper_grab_ik_rel_env_cfg import (
            DualPiperGrabIkRelEnvCfg,
        )
        env_cfg = DualPiperGrabIkRelEnvCfg()
        use_ik = True
    elif args_cli.env == "visuomotor":
        from isaaclab_tasks.manager_based.dual_piper.pick_place.dual_piper_grab_ik_rel_visuomotor_env_cfg import (
            DualPiperGrabIkRelVisuomotorEnvCfg,
        )
        env_cfg = DualPiperGrabIkRelVisuomotorEnvCfg()
        use_ik = True

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    env = ManagerBasedRLEnv(cfg=env_cfg)

    print("=" * 72)
    print(f"  Environment: {type(env_cfg).__name__}")
    print(f"  num_envs={env_cfg.scene.num_envs}")
    print(f"  action dim={env.action_manager.action.shape[1]}")
    print("=" * 72)

    count = 0
    use_random = True

    while simulation_app.is_running():
        with torch.inference_mode():
            if count % 500 == 0 or env.termination_manager.terminated.any():
                env.reset()
                print(f"[{count:5d}] Reset")

            if use_random:
                if use_ik:
                    actions = 0.02 * torch.randn(env.action_manager.action.shape, device=env.device)
                else:
                    actions = 0.01 * torch.randn(env.action_manager.action.shape, device=env.device)
            else:
                actions = torch.zeros(env.action_manager.action.shape, device=env.device)

            env.step(actions)
            count += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
