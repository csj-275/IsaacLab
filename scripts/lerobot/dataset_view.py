# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visualize joint action/state curves of a LeRobot (v3.0) dataset.

For each requested episode, plots every action/state dimension (e.g. joint1-8) in a single figure:
action as solid lines and observation.state as dashed lines.

Usage:
    # plot a single episode
    python dataset_view.py --dataset_dir datasets/lerobot/D-SIM-PIPER-GRAB-0702-N50-K-V1 --episode 1

    # plot episodes 1 to 10 (inclusive range), one figure per episode
    python dataset_view.py --dataset_dir datasets/lerobot/D-SIM-PIPER-GRAB-0702-N50-K-V1 --episode 1 10
"""

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot joint action/state curves of a LeRobot dataset.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the LeRobot dataset root directory.")
    parser.add_argument(
        "--episode",
        type=int,
        nargs="+",
        required=True,
        metavar="EP",
        help="Episode index to plot. One value plots a single episode; two values plot the inclusive range.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save figures. Defaults to logs/dataset_view/<dataset_name> at the repo root.",
    )
    return parser.parse_args()


def resolve_episodes(episode_arg: list[int]) -> list[int]:
    """Expand the --episode argument into a list of episode indices."""
    if len(episode_arg) == 1:
        return episode_arg
    if len(episode_arg) == 2:
        start, end = episode_arg
        if start > end:
            raise ValueError(f"Invalid episode range: {start} > {end}")
        return list(range(start, end + 1))
    # more than two values: treat as an explicit list
    return sorted(set(episode_arg))


def load_episodes_meta(dataset_dir: str) -> pd.DataFrame:
    """Load the per-episode metadata (v3.0: meta/episodes/chunk-*/file-*.parquet)."""
    meta_files = sorted(glob.glob(os.path.join(dataset_dir, "meta", "episodes", "chunk-*", "file-*.parquet")))
    if not meta_files:
        raise FileNotFoundError(f"No episode metadata found under {os.path.join(dataset_dir, 'meta', 'episodes')}")
    return pd.concat([pd.read_parquet(f) for f in meta_files], ignore_index=True)


def load_episode_data(dataset_dir: str, info: dict, ep_meta: pd.Series) -> pd.DataFrame:
    """Load the data frames of a single episode from its parquet file."""
    data_path = info["data_path"].format(
        chunk_index=int(ep_meta["data/chunk_index"]), file_index=int(ep_meta["data/file_index"])
    )
    df = pd.read_parquet(os.path.join(dataset_dir, data_path))
    df = df[df["episode_index"] == int(ep_meta["episode_index"])].sort_values("frame_index")
    return df


def plot_episode(df: pd.DataFrame, episode_index: int, joint_names: list[str], output_path: str) -> None:
    """Plot all joint dims: action (solid) vs state (dashed) for one episode and save the figure."""
    action = np.stack(df["action"].to_numpy())
    state = np.stack(df["observation.state"].to_numpy())
    t = df["timestamp"].to_numpy()

    num_joints = len(joint_names)
    ncols = 3
    nrows = (num_joints + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharex=True, squeeze=False)
    fig.suptitle(f"Episode {episode_index} — action (solid) vs state (dashed)", fontsize=14)

    for j in range(num_joints):
        ax = axes.flat[j]
        ax.plot(t, action[:, j], linestyle="-", color="tab:blue", label="action")
        ax.plot(t, state[:, j], linestyle="--", color="tab:orange", label="state")
        ax.set_title(joint_names[j])
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(loc="best", fontsize=8)
        if j >= num_joints - ncols:
            ax.set_xlabel("time (s)")
        if j % ncols == 0:
            ax.set_ylabel("position")
    # hide unused subplots
    for j in range(num_joints, nrows * ncols):
        axes.flat[j].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved episode {episode_index} ({len(df)} frames) -> {output_path}")


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_dir = args.output_dir or os.path.join(
        repo_root, "logs", "dataset_view", os.path.basename(os.path.normpath(dataset_dir))
    )
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(dataset_dir, "meta", "info.json")) as f:
        info = json.load(f)
    joint_names = info["features"]["action"]["names"]

    episodes_meta = load_episodes_meta(dataset_dir)
    available = set(episodes_meta["episode_index"].astype(int).tolist())

    for ep in resolve_episodes(args.episode):
        if ep not in available:
            print(f"[WARN] Episode {ep} not found in dataset (total: {info['total_episodes']}), skipping.")
            continue
        ep_meta = episodes_meta[episodes_meta["episode_index"] == ep].iloc[0]
        df = load_episode_data(dataset_dir, info, ep_meta)
        plot_episode(df, ep, joint_names, os.path.join(output_dir, f"episode_{ep:03d}.png"))


if __name__ == "__main__":
    main()
