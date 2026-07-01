# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Replay an existing HDF5 dataset (collected with state-only IK-rel env) in a
visuomotor environment to render camera images, and save directly to LeRobot format.

This bridges the gap between a state-only collected dataset and a multimodal
(image + state) LeRobot dataset by replaying recorded actions in a visuomotor
env that has cameras attached.

To ensure faithful replay, the script restores the initial joint state from
each source episode before stepping through the recorded actions.

Input:  Annotated HDF5 file from a state-only IK-rel environment.
Output: LeRobot dataset (parquet + MP4 videos).

Usage::

    ./isaaclab.sh -p \\
      scripts/lerobot/replay_to_lerobot.py \\
      --input_file datasets/hdf5/[0629]annotated_piper_dataset_K.hdf5 \\
      --output_dir datasets/lerobot/[0629]piper_dataset_K \\
      --visuomotor_task Isaac-Piper-Grab-IK-Rel-Visuomotor-v1 \\
      --headless --enable_cameras --device cuda:0 --fps 30
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import logging
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Replay an HDF5 dataset in a visuomotor environment and save as LeRobot format."
)
parser.add_argument(
    "--input_file", type=str, required=True,
    help="Path to the source annotated HDF5 dataset file.",
)
parser.add_argument(
    "--output_dir", type=str, required=True,
    help="Output directory for the LeRobot dataset.",
)
parser.add_argument(
    "--visuomotor_task", type=str, default="Isaac-Piper-Grab-IK-Rel-Visuomotor-v1",
    help="Name of the visuomotor Gym task to use for replay + rendering.",
)
parser.add_argument(
    "--fps", type=int, default=30,
    help="Frames-per-second for video encoding and metadata (default: 30).",
)
parser.add_argument(
    "--num_envs", type=int, default=1,
    help="Number of environments (should be 1 for deterministic replay).",
)
parser.add_argument(
    "--max_episodes", type=int, default=None,
    help="Maximum number of episodes to replay (default: all).",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import h5py
import torch

from isaaclab.managers.recorder_manager import DatasetExportMode
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.utils.datasets.lerobot_dataset_file_handler import LeRobotDatasetFileHandler
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import isaaclab_tasks  # noqa: F401

logger = logging.getLogger(__name__)


def _read_hdf5_metadata(input_path: str) -> dict:
    """Read top-level metadata from the HDF5 file."""
    with h5py.File(input_path, "r") as f:
        data = f["data"]
        env_args = data.attrs.get("env_args", "{}")
        if isinstance(env_args, bytes):
            env_args = env_args.decode("utf-8")
        import json
        return {
            "total_steps": int(data.attrs["total"]),
            "env_args": json.loads(env_args) if isinstance(env_args, str) else env_args,
            "episode_names": sorted(
                [k for k in data.keys() if k.startswith("demo_")],
                key=lambda x: int(x.split("_")[1]),
            ),
        }


def _load_episode_data(input_path: str, episode_name: str) -> dict:
    """Load a single episode's data from the HDF5 file.

    Returns a dict with:
        - actions: (T, A) tensor on CPU
        - num_samples: int
        - success: bool
        - initial_joint_pos: (1, J) tensor or None
        - initial_joint_vel: (1, J) tensor or None
    """
    with h5py.File(input_path, "r") as f:
        grp = f["data"][episode_name]
        actions = torch.from_numpy(grp["actions"][:]).float()
        num_samples = int(grp.attrs["num_samples"])
        success = bool(grp.attrs.get("success", True))

        # Extract initial joint state from the recorded initial_state group
        initial_joint_pos = None
        initial_joint_vel = None
        if "initial_state" in grp:
            istate = grp["initial_state"]
            if "articulation" in istate and "robot" in istate["articulation"]:
                robot_state = istate["articulation"]["robot"]
                if "joint_position" in robot_state:
                    initial_joint_pos = torch.from_numpy(robot_state["joint_position"][:]).float()
                if "joint_velocity" in robot_state:
                    initial_joint_vel = torch.from_numpy(robot_state["joint_velocity"][:]).float()

    return {
        "actions": actions,
        "num_samples": num_samples,
        "success": success,
        "initial_joint_pos": initial_joint_pos,
        "initial_joint_vel": initial_joint_vel,
    }


def _restore_initial_joint_state(env, initial_joint_pos, initial_joint_vel):
    """Write the source episode's initial joint state into the simulator.

    Args:
        env: The unwrapped environment.
        initial_joint_pos: (1, J) tensor of joint positions.
        initial_joint_vel: (1, J) tensor of joint velocities (can be None).
    """
    robot = env.scene["robot"]
    joint_pos = initial_joint_pos.to(env.device)
    if initial_joint_vel is not None:
        joint_vel = initial_joint_vel.to(env.device)
    else:
        joint_vel = torch.zeros_like(joint_pos)

    # Write joint state directly into physics buffers
    robot.write_joint_state_to_sim(
        position=joint_pos,
        velocity=joint_vel,
        joint_ids=slice(None),  # all joints
    )
    # Also set the PD target to the current position so the robot doesn't
    # snap back to the previous target
    robot.set_joint_position_target(joint_pos, joint_ids=slice(None))
    # Update the robot data buffers
    robot.write_data_to_sim()

    logger.info(
        f"Restored initial joint state: pos={joint_pos.tolist()}"
    )


def main():
    input_file = os.path.abspath(args_cli.input_file)
    output_dir = os.path.abspath(args_cli.output_dir)
    output_dir_name = os.path.basename(output_dir)

    # ------------------------------------------------------------------
    # 1. Read HDF5 metadata
    # ------------------------------------------------------------------
    meta = _read_hdf5_metadata(input_file)
    episode_names = meta["episode_names"]
    if args_cli.max_episodes is not None:
        episode_names = episode_names[: args_cli.max_episodes]

    logger.info(f"Source dataset: {len(episode_names)} episodes, {meta['total_steps']} total steps")
    logger.info(f"Source env: {meta['env_args'].get('env_name', 'unknown')}")

    # ------------------------------------------------------------------
    # 2. Build env config with LeRobot recorder
    # ------------------------------------------------------------------
    task_name = args_cli.visuomotor_task.split(":")[-1]
    env_cfg = parse_env_cfg(task_name, device=args_cli.device, num_envs=args_cli.num_envs)

    # Disable terminations during replay (we replay the full trajectory)
    env_cfg.terminations = None
    # Keep observations as a dict so the recorder can separate state vs images
    env_cfg.observations.policy.concatenate_terms = False

    # Disable domain randomization events for faithful replay.
    # We need the objects at their default positions so the initial state
    # restoration (via joint positions) lands in a consistent scene.
    if hasattr(env_cfg, "events"):
        # Clear all events so no randomization occurs on reset
        import dataclasses
        for field in dataclasses.fields(env_cfg.events):
            try:
                delattr(env_cfg.events, field.name)
            except (AttributeError, TypeError):
                pass
        logger.info("Cleared all domain randomization events.")

    # Configure LeRobot recorder
    lerobot_recorder_cfg = ActionStateRecorderManagerCfg()
    lerobot_recorder_cfg.dataset_file_handler_class_type = LeRobotDatasetFileHandler
    lerobot_recorder_cfg.dataset_export_dir_path = os.path.dirname(output_dir)
    lerobot_recorder_cfg.dataset_filename = output_dir_name
    lerobot_recorder_cfg.dataset_export_mode = DatasetExportMode.EXPORT_ALL

    env_cfg.recorders = lerobot_recorder_cfg

    # ------------------------------------------------------------------
    # 3. Create the visuomotor environment
    # ------------------------------------------------------------------
    logger.info(f"Creating visuomotor environment: {task_name}")
    env = gym.make(task_name, cfg=env_cfg)

    # Apply FPS setting to the LeRobot file handler
    rm = env.unwrapped.recorder_manager
    handler = getattr(rm, "_dataset_file_handler", None)
    if handler is not None and isinstance(handler, LeRobotDatasetFileHandler):
        handler.fps = args_cli.fps

    # ------------------------------------------------------------------
    # 4. Replay each episode
    # ------------------------------------------------------------------
    # Determine the action dimension the environment expects
    action_dim = env.unwrapped.action_manager.action.shape[-1]
    logger.info(f"Environment action dim: {action_dim}")

    total_replayed = 0

    for ep_idx, ep_name in enumerate(episode_names):
        ep_data = _load_episode_data(input_file, ep_name)
        actions = ep_data["actions"]  # (T, source_action_dim)
        num_samples = ep_data["num_samples"]
        source_action_dim = actions.shape[-1]

        logger.info(
            f"[{ep_idx + 1}/{len(episode_names)}] {ep_name}: "
            f"{num_samples} steps, source_action_dim={source_action_dim}, "
            f"has_initial_state={ep_data['initial_joint_pos'] is not None}"
        )

        # Handle action dimension mismatch
        if source_action_dim < action_dim:
            logger.warning(
                f"Source action dim ({source_action_dim}) < env action dim ({action_dim}). "
                f"Padding with zeros."
            )
            pad = torch.zeros(num_samples, action_dim - source_action_dim)
            actions = torch.cat([actions, pad], dim=-1)
        elif source_action_dim > action_dim:
            logger.warning(
                f"Source action dim ({source_action_dim}) > env action dim ({action_dim}). "
                f"Truncating."
            )
            actions = actions[:, :action_dim]

        # ---- Reset environment ----
        obs, info = env.reset()

        # ---- Restore initial joint state from source episode ----
        if ep_data["initial_joint_pos"] is not None:
            _restore_initial_joint_state(
                env.unwrapped,
                ep_data["initial_joint_pos"],
                ep_data["initial_joint_vel"],
            )

        # ---- Step through the episode ----
        for t in range(num_samples):
            action = actions[t].unsqueeze(0).to(env.unwrapped.device)
            obs, reward, terminated, truncated, info = env.step(action)

        total_replayed += num_samples

        # Export current episode by triggering the next reset.
        # The recorder exports on record_pre_reset() which is called inside env.reset().
        if ep_idx < len(episode_names) - 1:
            env.reset()

    # ------------------------------------------------------------------
    # 5. Export the final episode and write metadata
    # ------------------------------------------------------------------
    rm = env.unwrapped.recorder_manager
    logger.info("Exporting final episode...")
    rm.export_episodes()

    handler = getattr(rm, "_dataset_file_handler", None)
    if handler is not None and isinstance(handler, LeRobotDatasetFileHandler):
        logger.info("Flushing and closing LeRobot handler...")
        handler.flush()
        handler.close()

    env.close()

    logger.info(f"Done! Replayed {total_replayed} steps across {len(episode_names)} episodes.")
    logger.info(f"Output: {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
    # Close sim app
    simulation_app.close()
