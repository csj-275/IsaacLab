#!/usr/bin/env python3
"""Replay a training episode's actions in the eval environment and compare states.

Usage:
    export CUDA_VISIBLE_DEVICES=3
    ./isaaclab.sh -p scripts/lerobot/replay_action_test.py \\
        --dataset-dir ./datasets/lerobot/D-SIM-PIPER-GRAB-0702-N100-K-V1 \\
        --episode 5 --max-steps 200 \\
        --video ./logs/eval_videos --plot-dir ./logs/eval_plots
"""

import argparse
import glob, json, os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-dir", type=str, required=True)
parser.add_argument("--task", type=str, default="Isaac-Piper-Grab-IK-Rel-Visuomotor-v1-A")
parser.add_argument("--episode", type=int, default=0)
parser.add_argument("--max-steps", type=int, default=200)
parser.add_argument("--video", type=str, default=None)
parser.add_argument("--plot-dir", type=str, default="logs/eval_plots")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab_tasks  # noqa
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

print(f"Loading dataset: {args_cli.dataset_dir}")

# Load dataset
d = args_cli.dataset_dir
with open(os.path.join(d, "meta", "info.json")) as f:
    info = json.load(f)

meta_files = sorted(glob.glob(os.path.join(d, "meta", "episodes", "chunk-*", "file-*.parquet")))
ep_meta = pd.concat([pd.read_parquet(f) for f in meta_files], ignore_index=True)

data_files = sorted(glob.glob(os.path.join(d, "data", "chunk-*", "file-*.parquet")))
all_data = pd.concat([pd.read_parquet(f) for f in data_files], ignore_index=True)

ep_num = args_cli.episode
ep_data = all_data[all_data["episode_index"] == ep_num].sort_values("frame_index")
print(f"Episode {ep_num}: {len(ep_data)} frames")

# Extract actions and states
gt_actions = np.stack(ep_data["action"].values)  # (T, 7)
gt_states = np.stack(ep_data["observation.state"].values)  # (T, 7)

action_dim = gt_actions.shape[1]
print(f"Action dim: {action_dim}, State dim: {gt_states.shape[1]}")

# Create eval env
env_cfg = parse_env_cfg(task_name=args_cli.task, device=args_cli.device, num_envs=1)

from isaaclab.envs.mdp.actions.actions_cfg import AbsBinaryJointPositionActionCfg
env_cfg.actions.gripper_action = AbsBinaryJointPositionActionCfg(
    asset_name="robot",
    joint_names=["joint7", "joint8"],
    open_command_expr={"joint7": 0.05, "joint8": -0.05},
    close_command_expr={"joint7": -0.05, "joint8": 0.05},
    threshold=0.03,
    positive_threshold=True,
)

if action_dim == 8:
    from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
    env_cfg.actions.arm_action = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint[1-6]"],
        scale=1.0,
        use_default_offset=False,
    )
    env_cfg.actions.gripper_action = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint7", "joint8"],
        scale=1.0,
        use_default_offset=False,
    )

env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
print(f"Env created, action_space={env.action_space}")

# Setup video
video_dir = Path(args_cli.video) if args_cli.video else None
if video_dir:
    video_dir.mkdir(parents=True, exist_ok=True)
plot_dir = Path(args_cli.plot_dir)
plot_dir.mkdir(parents=True, exist_ok=True)

max_steps = min(args_cli.max_steps, len(gt_actions))
episode_frames = [] if video_dir else None
observed_states = []
observed_joint_targets = []

obs, _ = env.reset()
print(f"Initial obs joint_pos_7d: {obs['policy']['joint_pos_7d'].squeeze().tolist()}")

for step in range(max_steps):
    # Capture frame
    if episode_frames is not None:
        front = obs["policy"].get("table_cam")
        if front is not None:
            f = front.squeeze(0).cpu().numpy().astype(np.uint8)
            episode_frames.append(f)

    # Get GT action and feed to env
    gt_action = gt_actions[step]
    env_action = torch.from_numpy(gt_action).float().reshape(env.action_space.shape).to(env.device)
    obs, reward, terminated, truncated, info = env.step(env_action)

    # Record state
    obs_state = obs["policy"]["joint_pos_7d"].squeeze(0).cpu().numpy()
    observed_states.append(obs_state)

    # Record joint targets
    robot = env.scene["robot"]
    jt = robot.data.joint_pos_target[0].cpu().numpy()
    observed_joint_targets.append(jt)

    if step % 20 == 0 or step == max_steps - 1:
        gt_s = gt_states[step]
        print(f"Step {step}: gt_state={np.array2string(gt_s, precision=3, suppress_small=True)}, "
              f"obs_state={np.array2string(obs_state, precision=3, suppress_small=True)}")

    if bool(terminated.item()) or bool(truncated.item()):
        print(f"Episode ended at step {step}")
        break

observed_states = np.array(observed_states)

# Compare states
print(f"\n{'='*60}")
state_diff = np.abs(observed_states[:max_steps] - gt_states[:max_steps])
mean_diff = state_diff.mean(axis=0)
max_diff = state_diff.max(axis=0)

joint_names = ["j1", "j2", "j3", "j4", "j5", "j6", "grip"]
for j, name in enumerate(joint_names):
    print(f"  {name}: mean|diff|={mean_diff[j]:.4f}, max|diff|={max_diff[j]:.4f}")

# Plot
fig, axes = plt.subplots(4, 2, figsize=(16, 14))
axes = axes.flatten()
for j in range(min(7, action_dim)):
    ax = axes[j]
    steps = np.arange(max_steps)
    ax.plot(steps, gt_states[:max_steps, j], "-", linewidth=1.5, label="gt_state", color="C0")
    ax.plot(steps, observed_states[:max_steps, j], "--", linewidth=1.5, label="obs_state", color="C1")
    ax.set_title(f"{joint_names[j]} (mean|err|={mean_diff[j]:.4f})")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
axes[7].set_visible(False)
fig.suptitle(f"Episode {ep_num} — GT action replay in V1-A env", fontsize=14)
fig.tight_layout()
plot_path = plot_dir / f"replay_ep{ep_num:03d}.png"
fig.savefig(str(plot_path), dpi=100)
plt.close(fig)
print(f"Plot saved: {plot_path}")

# Save video
if episode_frames:
    import cv2
    vpath = video_dir / f"replay_ep{ep_num:03d}.mp4"
    h, w = episode_frames[0].shape[:2]
    writer = cv2.VideoWriter(str(vpath), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    for f in episode_frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"Video saved: {vpath}")

env.close()
print("Done.")
