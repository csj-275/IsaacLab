#!/usr/bin/env python3
"""
对比两个数据集的 14D joint position 数据：
  - Fig 1: 前 7 维 (current joint positions)
  - Fig 2: 后 7 维 (target joint positions)

用法:
    python scripts/lerobot/compare_datasets_14d.py \
        --dataset-a datasets/lerobot/SIM-PIPER-GRAB-0702-N100-K-V1 \
        --dataset-b datasets/lerobot/D-SIM-PIPER-GRAB-0702-N100-K-V1 \
        --episode 0

数据集 A (SIM-*, source format):
    observation.state: 14D [current(7) | target(7)]

数据集 B (D-SIM-*, converted with --use-state-as-action):
    observation.state: 7D (current)
    action:           7D (target)
    → 拼接后: [obs.state(:7) | action(:7)] = 14D
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]


def load_dataset_a(src_dir: Path) -> pd.DataFrame:
    """加载 source-format 数据集 (SIM-*)."""
    parquet_files = sorted(src_dir.glob("data/chunk-*/file-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files in {src_dir}")
    dfs = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(dfs, ignore_index=True)
    df.sort_values(["episode_index", "frame_index"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def load_dataset_b(src_dir: Path) -> pd.DataFrame:
    """加载 converted LeRobot 数据集 (D-SIM-*)."""
    return load_dataset_a(src_dir)


def get_14d_a(df: pd.DataFrame, ep_idx: int) -> np.ndarray:
    """从数据集 A 提取 14D: observation.state 即 [current(7) | target(7)]."""
    mask = df["episode_index"] == ep_idx
    ep = df[mask].sort_values("frame_index")
    return np.array([x for x in ep["observation.state"].values])


def get_14d_b(df: pd.DataFrame, ep_idx: int) -> np.ndarray:
    """从数据集 B 提取 14D: concat(observation.state, action) = [current(7) | target(7)]. """
    mask = df["episode_index"] == ep_idx
    ep = df[mask].sort_values("frame_index")
    obs = np.array([x for x in ep["observation.state"].values])
    act = np.array([x for x in ep["action"].values])
    return np.concatenate([obs, act], axis=1)


def plot_comparison(data_a_14d: np.ndarray, data_b_14d: np.ndarray, label_a: str, label_b: str):
    """绘制对比图: Fig1=前7维, Fig2=后7维，每维一个子图."""

    timesteps_a = np.arange(len(data_a_14d))
    timesteps_b = np.arange(len(data_b_14d))

    for fig_idx, (dim_start, dim_label) in enumerate([(0, "first_7d"), (7, "last_7d")]):
        fig, axes = plt.subplots(4, 2, figsize=(16, 12))
        fig.suptitle(f"{dim_label}: {'Current (joint1~gripper)' if dim_start == 0 else 'Target (joint1~gripper)'}",
                     fontsize=14, fontweight="bold")
        axes = axes.flatten()

        for i in range(7):
            dim = dim_start + i
            ax = axes[i]
            ax.plot(timesteps_a, data_a_14d[:, dim], color="red", linestyle="-",
                    linewidth=1.0, alpha=0.8, label=f"{label_a}")
            ax.plot(timesteps_b, data_b_14d[:, dim], color="blue", linestyle="--",
                    linewidth=1.0, alpha=0.8, label=f"{label_b}")
            ax.set_title(f"dim{dim}: {_JOINT_NAMES[i]}")
            ax.set_xlabel("Frame")
            ax.set_ylabel("Joint position")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        # 隐藏多余子图
        if len(axes) > 7:
            axes[7].set_visible(False)

        plt.tight_layout()

        out_path = f"compare_{dim_label}.png"
        fig.savefig(out_path, dpi=150)
        print(f"Saved: {out_path}")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare 14D state between two datasets")
    parser.add_argument("--dataset-a", type=str, required=True,
                        help="Path to dataset A (source format, SIM-*)")
    parser.add_argument("--dataset-b", type=str, required=True,
                        help="Path to dataset B (converted format, D-SIM-*)")
    parser.add_argument("--episode", type=int, default=0,
                        help="Episode index to plot (default: 0)")
    args = parser.parse_args()

    dir_a = Path(args.dataset_a)
    dir_b = Path(args.dataset_b)

    print(f"Loading dataset A: {dir_a}")
    df_a = load_dataset_a(dir_a)
    print(f"  shape={df_a.shape}, episodes={df_a['episode_index'].nunique()}")

    print(f"Loading dataset B: {dir_b}")
    df_b = load_dataset_b(dir_b)
    print(f"  shape={df_b.shape}, episodes={df_b['episode_index'].nunique()}")

    # Metadata labels
    label_a = dir_a.name
    label_b = dir_b.name

    ep = args.episode
    print(f"\nExtracting episode {ep}...")
    data_a = get_14d_a(df_a, ep)
    data_b = get_14d_b(df_b, ep)
    print(f"  A: {data_a.shape} (frames x 14)")
    print(f"  B: {data_b.shape} (frames x 14)")

    plot_comparison(data_a, data_b, label_a, label_b)

    print("\nDone.")


if __name__ == "__main__":
    main()
