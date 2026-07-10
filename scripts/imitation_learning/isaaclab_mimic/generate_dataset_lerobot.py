# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Data generation script that saves directly to LeRobot format.

Unlike ``generate_dataset.py`` which produces a large intermediate HDF5 file
that must be post-converted, this script writes episodes on-the-fly as parquet
files + MP4 videos — no intermediate HDF5, no disk-space explosion.

Output format matches the reference ``SIM-PIPER-GRAB-0604-N25-V1`` layout:
    - codebase_version: v3.0
    - Parquet: file-{idx:03d}.parquet, columns: action, observation.state,
      timestamp, frame_index, episode_index, index, task_index
    - Video: file-{idx:03d}.mp4 under videos/observation.{images,depths}.<cam>/
    - Meta: info.json, stats.json, tasks.parquet, episodes jsonl+parquet,
      recording_meta.json

Usage::

    # Headless on a specific GPU (recommended):
    DISPLAY= CUDA_VISIBLE_DEVICES=1 ./isaaclab.sh -p \\
      scripts/imitation_learning/isaaclab_mimic/generate_dataset_lerobot.py \\
      --task Isaac-Piper-Grab-IK-Rel-Visuomotor-Mimic-v1 \\
      --input_file ./datasets/simdata/V1/[0604]annotated_visuo_dataset.hdf5 \\
      --output_file ./datasets/simdata/V1/lerobot_generated_N10 \\
      --generation_num_trials 10 \\
      --headless --enable_cameras --device cuda:0 \\
      --fps 30

Camera keys are auto-mapped:
    - table_cam     → observation.images.front
    - wrist_cam     → observation.images.wrist
    - table_cam_depth → observation.depths.front
    - wrist_cam_depth → observation.depths.wrist
"""

"""Launch Isaac Sim Simulator first."""

# ---------------------------------------------------------------------------
# State key filter: only these obs keys are saved in observation.state.
# Set to None to include ALL non-image keys (default behavior).
# Example: ["actions"] → 7D state, ["actions", "joint_pos"] → 15D state.
# ---------------------------------------------------------------------------
STATE_KEY_FILTER = ["actions"]  # 7D: joint1-6 + gripper positions
# ---------------------------------------------------------------------------

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Generate demonstrations for Isaac Lab environments, saving directly to LeRobot format."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--generation_num_trials", type=int, help="Number of demos to be generated.", default=None
)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to instantiate for generating datasets."
)
parser.add_argument(
    "--input_file", type=str, default=None, required=True, help="File path to the source (annotated HDF5) dataset file."
)
parser.add_argument(
    "--output_file",
    type=str,
    default="./datasets/output_lerobot",
    help="Output directory for the LeRobot dataset (NOT an HDF5 file).",
)
parser.add_argument(
    "--pause_subtask",
    action="store_true",
    help="pause after every subtask during generation for debugging - only useful with render flag",
)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)
parser.add_argument(
    "--use_skillgen",
    action="store_true",
    default=False,
    help="use skillgen to generate motion trajectories",
)
parser.add_argument(
    "--fps",
    type=int,
    default=30,
    help="Frames-per-second for video encoding and metadata (default: 30).",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version
    # installed by IsaacLab and not the one installed by Isaac Sim.
    # pinocchio is required by the Pink IK controllers and the GR1T2 retargeter
    import pinocchio  # noqa: F401

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import asyncio
import inspect
import logging
import random

import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.utils.datasets.lerobot_dataset_file_handler import LeRobotDatasetFileHandler

import isaaclab_mimic.envs  # noqa: F401

if args_cli.enable_pinocchio:
    import isaaclab_mimic.envs.pinocchio_envs  # noqa: F401

from isaaclab_mimic.datagen.generation import env_loop, setup_async_generation, setup_env_config
from isaaclab_mimic.datagen.utils import get_env_name_from_dataset, setup_output_paths

import isaaclab_tasks  # noqa: F401

# import logger
logger = logging.getLogger(__name__)


def main():
    num_envs = args_cli.num_envs

    # Setup output paths and get env name
    output_dir, output_file_name = setup_output_paths(args_cli.output_file)
    task_name = args_cli.task
    if task_name:
        task_name = args_cli.task.split(":")[-1]
    env_name = task_name or get_env_name_from_dataset(args_cli.input_file)

    # ------------------------------------------------------------------
    # Build a RecorderManagerCfg that uses the LeRobot handler
    # ------------------------------------------------------------------
    lerobot_recorder_cfg = ActionStateRecorderManagerCfg()
    lerobot_recorder_cfg.dataset_file_handler_class_type = LeRobotDatasetFileHandler

    # Configure environment
    env_cfg, success_term = setup_env_config(
        env_name=env_name,
        output_dir=output_dir,
        output_file_name=output_file_name,
        num_envs=num_envs,
        device=args_cli.device,
        generation_num_trials=args_cli.generation_num_trials,
        recorder_cfg=lerobot_recorder_cfg,
    )

    # Create environment
    env = gym.make(env_name, cfg=env_cfg).unwrapped

    if not isinstance(env, ManagerBasedRLMimicEnv):
        raise ValueError("The environment should be derived from ManagerBasedRLMimicEnv")

    # Apply FPS setting to the LeRobot file handler
    _apply_handler_settings(env, args_cli)

    # Check if the mimic API from this environment contains deprecated signatures
    if "action_noise_dict" not in inspect.signature(env.target_eef_pose_to_action).parameters:
        logger.warning(
            f'The "noise" parameter in the "{env_name}" environment\'s mimic API "target_eef_pose_to_action", '
            "is deprecated. Please update the API to take action_noise_dict instead."
        )

    # Set seed for generation
    random.seed(env.cfg.datagen_config.seed)
    np.random.seed(env.cfg.datagen_config.seed)
    torch.manual_seed(env.cfg.datagen_config.seed)

    # Reset before starting
    env.reset()

    motion_planners = None
    if args_cli.use_skillgen:
        from isaaclab_mimic.motion_planners.curobo.curobo_planner import CuroboPlanner
        from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg

        # Create one motion planner per environment
        motion_planners = {}
        for env_id in range(num_envs):
            print(f"Initializing motion planner for environment {env_id}")
            planner_config = CuroboPlannerCfg.from_task_name(env_name)

            if env_id != 0:
                planner_config.visualize_spheres = False
                planner_config.visualize_plan = False

            motion_planners[env_id] = CuroboPlanner(
                env=env,
                robot=env.scene["robot"],
                config=planner_config,
                env_id=env_id,
            )

        env.cfg.datagen_config.use_skillgen = True

    # Setup and run async data generation
    async_components = setup_async_generation(
        env=env,
        num_envs=args_cli.num_envs,
        input_file=args_cli.input_file,
        success_term=success_term,
        pause_subtask=args_cli.pause_subtask,
        motion_planners=motion_planners,
    )

    try:
        data_gen_tasks = asyncio.ensure_future(asyncio.gather(*async_components["tasks"]))
        env_loop(
            env,
            async_components["reset_queue"],
            async_components["action_queue"],
            async_components["info_pool"],
            async_components["event_loop"],
        )
    except asyncio.CancelledError:
        print("Tasks were cancelled.")
    finally:
        # Cancel all async tasks when env_loop finishes
        data_gen_tasks.cancel()
        try:
            async_components["event_loop"].run_until_complete(data_gen_tasks)
        except asyncio.CancelledError:
            print("Remaining async tasks cancelled and cleaned up.")
        except Exception as e:
            print(f"Error cancelling remaining async tasks: {e}")
        # Cleanup of motion planners and their visualizers
        if motion_planners is not None:
            for env_id, planner in motion_planners.items():
                if getattr(planner, "plan_visualizer", None) is not None:
                    print(f"Closing plan visualizer for environment {env_id}")
                    planner.plan_visualizer.close()
                    planner.plan_visualizer = None
            motion_planners.clear()


def _apply_handler_settings(env, args_cli) -> None:
    """Apply user-specified settings to the LeRobot dataset file handler."""
    try:
        rm = env.recorder_manager
    except AttributeError:
        return

    handler = getattr(rm, "_dataset_file_handler", None)
    if handler is None or not isinstance(handler, LeRobotDatasetFileHandler):
        return

    handler.fps = args_cli.fps
    handler.state_key_filter = STATE_KEY_FILTER


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
    # Close sim app
    simulation_app.close()
