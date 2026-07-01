#!/usr/bin/env python3
# Copyright (c) 2024-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Replay annotated HDF5 demonstrations in a visuomotor environment and save
directly to LeRobot format (parquet + MP4 videos).

This replaces the old two-step workflow (replay → parquet → convert) with a
single step.  The LeRobot file handler writes episodes on-the-fly during replay.

Output layout (LeRobot v3.0):
    data/chunk-000/file-{ep:03d}.parquet
    videos/observation.images.{front,wrist}/chunk-000/file-{ep:03d}.mp4
    meta/info.json, meta/stats.json, meta/tasks.parquet, meta/episodes/…

Usage (inside Docker container)::

    ./isaaclab.sh -p scripts/lerobot/convert_to_lerobot.py \\
        --task Isaac-Piper-Grab-IK-Rel-Visuomotor-v1 \\
        --input ./datasets/hdf5/[0629]annotated_piper_dataset_K.hdf5 \\
        --output ./datasets/lerobot/piper_grab_replayed \\
        --fps 30 \\
        --headless --enable_cameras --device cuda:0

    # Only replay specific episodes:
    ./isaaclab.sh -p scripts/lerobot/convert_to_lerobot.py \\
        --task Isaac-Piper-Grab-IK-Rel-Visuomotor-v1 \\
        --input ./datasets/hdf5/[0629]annotated_piper_dataset_K.hdf5 \\
        --output ./datasets/lerobot/piper_grab_replayed \\
        --select_episodes 0 5 10 \\
        --headless --enable_cameras --device cuda:0
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Replay HDF5 demos in a visuomotor env and save directly to LeRobot format."
)
parser.add_argument("--task", type=str, required=True, help="Isaac Lab task name (visuomotor env).")
parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="Path to the annotated HDF5 dataset file.",
)
parser.add_argument(
    "--output",
    type=str,
    required=True,
    help="Output directory for the LeRobot dataset (will be created if needed).",
)
parser.add_argument("--fps", type=int, default=30, help="FPS for video encoding and metadata (default: 30).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to replay episodes.")
parser.add_argument(
    "--select_episodes",
    type=int,
    nargs="+",
    default=[],
    help="Specific episode indices to replay. Omit to replay all.",
)
parser.add_argument(
    "--skip_failed",
    action="store_true",
    default=False,
    help="Only save successful episodes (check via env subtask_terms / terminations).",
)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import logging
from pathlib import Path

import gymnasium as gym
import torch

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.utils.datasets import HDF5DatasetFileHandler, EpisodeData
from isaaclab.utils.datasets.lerobot_dataset_file_handler import LeRobotDatasetFileHandler

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    # ------------------------------------------------------------------
    # Load HDF5 dataset
    # ------------------------------------------------------------------
    input_path = Path(args_cli.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(str(input_path))
    episode_count = dataset_file_handler.get_num_episodes()

    if episode_count == 0:
        print("No episodes found in the dataset.")
        exit()

    episode_indices_to_replay = args_cli.select_episodes
    if len(episode_indices_to_replay) == 0:
        episode_indices_to_replay = list(range(episode_count))

    task_name = args_cli.task.split(":")[-1]
    num_envs = args_cli.num_envs

    # ------------------------------------------------------------------
    # Configure environment with LeRobot recorder
    # ------------------------------------------------------------------
    output_dir = Path(args_cli.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Task: {task_name}")
    logger.info(f"Input: {input_path} ({episode_count} episodes)")
    logger.info(f"Output: {args_cli.output} (dataset: {output_dir.name})")
    logger.info(f"Episodes to replay: {len(episode_indices_to_replay)}")

    env_cfg = parse_env_cfg(task_name, device=args_cli.device, num_envs=num_envs)

    # --- Extract success check before disabling terminations ---
    # 优先使用 subtask_terms（如 placed_1），fallback 到 terminations.success
    success_check_fn = None
    if hasattr(env_cfg, "terminations") and hasattr(env_cfg.terminations, "success"):
        success_check_fn = env_cfg.terminations.success

    # Disable terminations — we want to replay the full episode without early termination
    env_cfg.terminations = {}

    # Setup LeRobot recorder
    lerobot_recorder_cfg = ActionStateRecorderManagerCfg()
    lerobot_recorder_cfg.dataset_file_handler_class_type = LeRobotDatasetFileHandler
    dataset_name = output_dir.name
    lerobot_recorder_cfg.dataset_export_dir_path = str(output_dir)
    lerobot_recorder_cfg.dataset_filename = dataset_name

    env_cfg.recorders = lerobot_recorder_cfg

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # Apply FPS setting to the LeRobot file handler
    try:
        handler = env.recorder_manager._dataset_file_handler
        if isinstance(handler, LeRobotDatasetFileHandler):
            handler.fps = args_cli.fps
            logger.info(f"LeRobot handler FPS set to {args_cli.fps}")
    except AttributeError:
        pass

    logger.info(f"Environment created: {task_name}")
    if success_check_fn is not None:
        logger.info(f"Success check: {success_check_fn.func.__name__}")
    logger.info(f"Skip failed episodes: {args_cli.skip_failed}")

    # ------------------------------------------------------------------
    # Replay episodes
    # ------------------------------------------------------------------
    idle_action = torch.zeros(env.action_space.shape)
    episode_names = list(dataset_file_handler.get_episode_names())
    replayed_episode_count = 0
    success_episodes = []
    failed_episodes = []
    current_episode_indices = [None] * num_envs
    episode_ended = [False] * num_envs

    env.reset()

    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running() and not simulation_app.is_exiting():
            env_episode_data_map = {i: EpisodeData() for i in range(num_envs)}
            first_loop = True
            has_next_action = True

            while has_next_action:
                actions = idle_action.clone()
                has_next_action = False

                for env_id in range(num_envs):
                    env_next_action = env_episode_data_map[env_id].get_next_action()
                    if env_next_action is None:
                        # --- Episode finished on this env ---
                        # Check success BEFORE loading next episode
                        just_finished_idx = current_episode_indices[env_id]
                        if just_finished_idx is not None and not episode_ended[env_id]:
                            episode_ended[env_id] = True
                            is_success = _check_episode_success(env, env_id, success_check_fn)
                            if is_success:
                                success_episodes.append(just_finished_idx)
                            else:
                                failed_episodes.append(just_finished_idx)
                            status = "✅" if is_success else "❌"
                            logger.info(
                                f"  {status} Episode #{just_finished_idx} "
                                f"({'SUCCESS' if is_success else 'FAILED'})"
                            )

                        # Load next episode
                        next_episode_index = None
                        while episode_indices_to_replay:
                            next_episode_index = episode_indices_to_replay.pop(0)
                            if next_episode_index < episode_count:
                                episode_ended[env_id] = False
                                break
                            next_episode_index = None

                        if next_episode_index is not None:
                            replayed_episode_count += 1
                            current_episode_indices[env_id] = next_episode_index
                            logger.info(
                                f"[{replayed_episode_count}/{episode_count}] "
                                f"Replaying episode #{next_episode_index} on env_{env_id}"
                            )
                            episode_data = dataset_file_handler.load_episode(
                                episode_names[next_episode_index], env.device
                            )
                            env_episode_data_map[env_id] = episode_data
                            initial_state = episode_data.get_initial_state()
                            env.reset_to(initial_state, torch.tensor([env_id], device=env.device), is_relative=True)
                            env_next_action = env_episode_data_map[env_id].get_next_action()
                            has_next_action = True
                        else:
                            continue
                    else:
                        has_next_action = True

                    actions[env_id] = env_next_action

                if first_loop:
                    first_loop = False
                else:
                    env.step(actions)

            break  # All episodes done

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total = len(success_episodes) + len(failed_episodes)
    logger.info(f"{'='*50}")
    logger.info(f"Replay complete: {total} episodes")
    logger.info(f"  ✅ Success: {len(success_episodes)}")
    logger.info(f"  ❌ Failed:  {len(failed_episodes)}")
    if total > 0:
        logger.info(f"  Success rate: {100 * len(success_episodes) / total:.1f}%")
    logger.info(f"LeRobot dataset saved to: {output_dir.resolve()}")

    if args_cli.skip_failed:
        _delete_failed_episodes(output_dir, failed_episodes)
        logger.info(f"Deleted {len(failed_episodes)} failed episodes (--skip_failed)")
        logger.info(f"Remaining: {len(success_episodes)} successful episodes")

    # Print output structure summary
    data_dir = output_dir / "data" / "chunk-000"
    video_dir = output_dir / "videos"
    if data_dir.exists():
        parquet_files = sorted(data_dir.glob("*.parquet"))
        logger.info(f"  Parquet files: {len(parquet_files)}")
    if video_dir.exists():
        for cam_dir in sorted(video_dir.glob("observation.images.*")):
            mp4_files = sorted((cam_dir / "chunk-000").glob("*.mp4"))
            logger.info(f"  {cam_dir.name}: {len(mp4_files)} videos")

    if failed_episodes and not args_cli.skip_failed:
        logger.info(f"Failed episodes: {sorted(failed_episodes)}")

    env.close()


def _check_episode_success(env, env_id: int, success_term) -> bool:
    """Check if the episode on env_id is successful.

    Priority:
    1. subtask_terms["placed_1"] in observation buffer
    2. success_term function (if available)
    """
    # 方法 1: subtask_terms（visuomotor 环境自带）
    obs_buf = getattr(env, "obs_buf", {})
    subtask_terms = obs_buf.get("subtask_terms", {})
    placed = subtask_terms.get("placed_1")
    if placed is not None:
        if isinstance(placed, torch.Tensor):
            return bool(placed[env_id].item()) if placed.numel() > env_id else False
        return bool(placed)

    # 方法 2: success termination function
    if success_term is not None:
        return bool(success_term.func(env, **success_term.params)[env_id])

    return False


def _delete_failed_episodes(output_dir: Path, failed_indices: list[int]) -> None:
    """Delete parquet and video files for failed episodes."""
    import shutil

    for ep_idx in failed_indices:
        # Parquet
        parquet_file = output_dir / "data" / "chunk-000" / f"file-{ep_idx:03d}.parquet"
        if parquet_file.exists():
            parquet_file.unlink()
            logger.debug(f"Deleted {parquet_file}")

        # Videos
        video_base = output_dir / "videos"
        if video_base.exists():
            for cam_dir in video_base.glob("observation.images.*"):
                mp4_file = cam_dir / "chunk-000" / f"file-{ep_idx:03d}.mp4"
                if mp4_file.exists():
                    mp4_file.unlink()
                    logger.debug(f"Deleted {mp4_file}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        simulation_app.close()
