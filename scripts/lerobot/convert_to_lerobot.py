#!/usr/bin/env python3
"""
将 isaaclab/datasets/simdata 中的 IsaacLab 数据转换为 LeRobot v3.0 格式。

用法:
    conda activate lerobot

    # 指定源数据集和输出目录
    python scripts/lerobot/convert_to_lerobot.py \
        --src-dir /home/chenshengjia/company/isaaclab/datasets/simdata/V1/SIM-PIPER-GRAB-0618-N100-IK-K-V1 \
        --output-dir /home/chenshengjia/company/isaaclab/datasets/lerobot/piper_grab_v1

    # 跳过视频
    python scripts/lerobot/convert_to_lerobot.py --skip-videos \
        --src-dir ... --output-dir ...

    # Docker 容器内
    python scripts/lerobot/convert_to_lerobot.py \
        --src-dir /workspace/isaaclab/datasets/simdata/V1/SIM-PIPER-GRAB-0618-N100-IK-K-V1 \
        --output-dir /workspace/isaaclab/datasets/lerobot/piper_grab_v1

数据格式:
  - action: 8维 (IK空间下 delta pose + gripper)
  - observation.state: 63维 (robot state + object states)
  - observation.images.front: 1280x720 @ 30fps
  - observation.images.wrist: 1280x720 @ 30fps
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset
from tqdm import tqdm

from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
from lerobot.datasets.feature_utils import create_empty_dataset_info
from lerobot.datasets.io_utils import write_episodes, write_info, write_stats, write_tasks
from lerobot.datasets.dataset_metadata import CODEBASE_VERSION
from lerobot.datasets.utils import (
    DEFAULT_DATA_PATH,
    DEFAULT_EPISODES_PATH,
    DEFAULT_VIDEO_PATH,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_features() -> dict:
    """构建 LeRobot features 字典."""
    return {
        "action": {
            "dtype": "float32",
            "shape": (8,),
            "type": "ACTION",
            "names": None,
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (63,),
            "type": "STATE",
            "names": None,
        },
        "observation.images.front": {
            "dtype": "video",
            "shape": (720, 1280, 3),
            "type": "VISUAL",
            "names": ["height", "width", "channel"],
        },
        "observation.images.wrist": {
            "dtype": "video",
            "shape": (720, 1280, 3),
            "type": "VISUAL",
            "names": ["height", "width", "channel"],
        },
    }


def read_isaaclab_data(data_dir: Path) -> pd.DataFrame:
    """读取所有 IsaacLab parquet 文件并合并为一个 DataFrame."""
    parquet_files = sorted(data_dir.glob("data/chunk-*/file-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
    logger.info(f"Found {len(parquet_files)} parquet files")

    dfs = []
    for pf in tqdm(parquet_files, desc="Reading parquet files"):
        df = pd.read_parquet(pf)
        dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)
    full_df.sort_values(["episode_index", "frame_index"], inplace=True)
    full_df.reset_index(drop=True, inplace=True)
    logger.info(f"Total frames: {len(full_df)}")
    logger.info(f"Episodes: {full_df['episode_index'].nunique()}")
    return full_df


def build_episodes_metadata(full_df: pd.DataFrame, output_dir: Path, features: dict) -> Dataset:
    """构建 episodes metadata (Parquet 格式)."""
    episodes_data = []
    video_keys = [k for k, v in features.items() if v["dtype"] == "video"]

    # 所有数据在一个 chunk-000 中，每个 episode 对应一个 file-XXX
    for ep_idx, group in tqdm(full_df.groupby("episode_index"), desc="Building episodes"):
        ep_idx = int(ep_idx)
        group = group.sort_values("frame_index")
        from_idx = int(group["index"].min())
        to_idx = int(group["index"].max())

        ep_entry = {
            "episode_index": ep_idx,
            "tasks": ["pick cube then grab bottle"],
            "length": len(group),
            "dataset_from_index": from_idx,
            "dataset_to_index": to_idx,
            "from_timestamp": float(group["timestamp"].min()),
            "to_timestamp": float(group["timestamp"].max()),
            "data/chunk_index": 0,
            "data/file_index": ep_idx,
        }

        # 视频文件: 每个 episode 对应 file-{ep_idx}.mp4
        ep_duration = float(group["timestamp"].max())
        for vk in video_keys:
            ep_entry[f"videos/{vk}/chunk_index"] = 0
            ep_entry[f"videos/{vk}/file_index"] = ep_idx
            ep_entry[f"videos/{vk}/from_timestamp"] = 0.0
            ep_entry[f"videos/{vk}/to_timestamp"] = ep_duration

        episodes_data.append(ep_entry)

    episodes_df = pd.DataFrame(episodes_data)
    return Dataset.from_pandas(episodes_df)


def save_data_parquet_files(full_df: pd.DataFrame, output_dir: Path) -> None:
    """将 IsaacLab 数据重新保存为 LeRobot 兼容的 parquet 文件。

    每 episode 一个 file-{ep_idx:03d}.parquet，放在 chunk-000 下。
    LeRobot v3.0 通过 datasets.Dataset.from_parquet() 读取，因此我们使用
    HuggingFace datasets 兼容的格式写入。
    跳过 observation.images.* (视频存在 mp4 中)。
    """
    data_pattern = DEFAULT_DATA_PATH  # "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"

    columns_to_save = ["action", "observation.state", "timestamp", "frame_index",
                       "episode_index", "index", "task_index"]

    for ep_idx, group in tqdm(full_df.groupby("episode_index"), desc="Saving data parquet"):
        ep_idx = int(ep_idx)
        group = group.sort_values("frame_index")
        ep_df = group[columns_to_save].copy()

        # 确保列类型正确
        ep_df["timestamp"] = ep_df["timestamp"].astype("float32")
        ep_df["frame_index"] = ep_df["frame_index"].astype("int64")
        ep_df["episode_index"] = ep_df["episode_index"].astype("int64")
        ep_df["index"] = ep_df["index"].astype("int64")
        ep_df["task_index"] = ep_df["task_index"].astype("int64")

        fpath = output_dir / data_pattern.format(chunk_index=0, file_index=ep_idx)
        Path(fpath).parent.mkdir(parents=True, exist_ok=True)
        ep_df.to_parquet(fpath, index=False)

    logger.info(f"Saved {full_df['episode_index'].nunique()} data parquet files to {output_dir / 'data'}")


def compute_and_save_stats(full_df: pd.DataFrame, output_dir: Path, features: dict) -> None:
    """计算并保存数据集统计信息（用于归一化）。"""
    logger.info("Computing dataset statistics...")

    # 一次处理一个 episode 来统计
    all_ep_stats = []
    video_keys = [k for k, v in features.items() if v["dtype"] == "video"]

    for ep_idx, group in tqdm(full_df.groupby("episode_index"), desc="Computing stats"):
        # 构建 episode buffer 格式: { key: np.array([frame_values...]) }
        episode_buffer = {
            "action": np.stack(group["action"].values),
            "observation.state": np.stack(group["observation.state"].values),
        }
        ep_stat = compute_episode_stats(episode_buffer, features)
        all_ep_stats.append(ep_stat)

    # 聚合所有 episode 的统计
    stats = aggregate_stats(all_ep_stats)
    write_stats(stats, output_dir)
    logger.info("Stats saved to meta/stats.json")


def link_video_files(src_dir: Path, output_dir: Path, full_df: pd.DataFrame,
                     features: dict, symlink: bool = False) -> None:
    """将视频文件链接/复制到 LeRobot 目录。

    IsaacLab 视频命名: file-{ep:03d}.mp4
    LeRobot 视频命名: file-{ep:03d}.mp4 (相同)
    """
    video_pattern = DEFAULT_VIDEO_PATH  # "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    video_keys = [k for k, v in features.items() if v["dtype"] == "video"]

    # 源视频目录映射
    src_video_dirs = {
        "observation.images.front": src_dir / "videos" / "observation.images.front" / "chunk-000",
        "observation.images.wrist": src_dir / "videos" / "observation.images.wrist" / "chunk-000",
    }

    for vk in video_keys:
        out_video_dir = output_dir / "videos" / vk / "chunk-000"
        out_video_dir.mkdir(parents=True, exist_ok=True)

        src_video_dir = src_video_dirs.get(vk)
        if src_video_dir is None or not src_video_dir.exists():
            logger.warning(f"Source video directory not found: {src_video_dir}, skipping {vk}")
            continue

        copy_func = shutil.copy2 if not symlink else _symlink_or_copy

        episodes = sorted(full_df["episode_index"].unique())
        for ep_idx in tqdm(episodes, desc=f"Linking videos for {vk}"):
            ep_idx = int(ep_idx)
            # IsaacLab: file-{ep:03d}.mp4
            src_file = src_video_dir / f"file-{ep_idx:03d}.mp4"
            # LeRobot: file-{ep:03d}.mp4
            dst_file = out_video_dir / f"file-{ep_idx:03d}.mp4"

            if src_file.exists() and not dst_file.exists():
                copy_func(src_file, dst_file)

        logger.info(f"Linked {len(episodes)} videos for {vk}")


def _symlink_or_copy(src: Path, dst: Path) -> None:
    """尝试 symlink，失败则复制。"""
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(description="Convert IsaacLab data to LeRobot v3.0 format")
    parser.add_argument(
        "--src-dir",
        type=str,
        default="/home/chenshengjia/company/isaaclab/datasets/simdata/V1/SIM-PIPER-GRAB-0618-N100-IK-K-V1",
        help="Source IsaacLab data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/chenshengjia/company/isaaclab/datasets/lerobot/piper_grab_v1",
        help="Output LeRobot dataset directory",
    )
    parser.add_argument("--symlink", action="store_true", help="Use symlinks instead of copying video files")
    parser.add_argument("--skip-videos", action="store_true", help="Skip video processing")
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    output_dir = Path(args.output_dir)

    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 读取 IsaacLab 数据
    logger.info("=" * 60)
    logger.info("Step 1: Reading IsaacLab data")
    logger.info("=" * 60)
    full_df = read_isaaclab_data(src_dir)

    # 2. 构建 features
    features = build_features()

    # 3. 创建 info.json
    logger.info("=" * 60)
    logger.info("Step 2: Creating meta/info.json")
    logger.info("=" * 60)
    info = create_empty_dataset_info(
        codebase_version=CODEBASE_VERSION,
        fps=30,
        features=features,
        use_videos=True,
        robot_type="piper",
        chunks_size=1000,
        data_files_size_in_mb=100,
        video_files_size_in_mb=200,
    )
    # 更新 total_episodes, total_frames, total_tasks
    num_episodes = full_df["episode_index"].nunique()
    num_frames = len(full_df)
    info.total_episodes = num_episodes
    info.total_frames = num_frames
    info.total_tasks = 1
    write_info(info, output_dir)
    logger.info(f"info.json written to {output_dir / 'meta/info.json'}")

    # 4. 构建并写入 episodes metadata
    logger.info("=" * 60)
    logger.info("Step 3: Building episodes metadata")
    logger.info("=" * 60)
    episodes_ds = build_episodes_metadata(full_df, output_dir, features)
    write_episodes(episodes_ds, output_dir)
    logger.info(f"Episodes written to {output_dir / DEFAULT_EPISODES_PATH.format(chunk_index=0, file_index=0)}")

    # 5. 写入 tasks
    logger.info("=" * 60)
    logger.info("Step 4: Writing tasks")
    logger.info("=" * 60)
    tasks_ds = Dataset.from_pandas(pd.DataFrame({
        "task_index": [0],
        "task": ["pick cube then grab bottle"],
    }))
    write_tasks(tasks_ds, output_dir)
    logger.info(f"Tasks written to {output_dir / 'meta/tasks.parquet'}")

    # 6. 保存数据 parquet 文件
    logger.info("=" * 60)
    logger.info("Step 5: Saving data parquet files")
    logger.info("=" * 60)
    save_data_parquet_files(full_df, output_dir)

    # 7. 计算并保存统计信息
    logger.info("=" * 60)
    logger.info("Step 6: Computing stats")
    logger.info("=" * 60)
    compute_and_save_stats(full_df, output_dir, features)

    # 8. 视频文件
    if not args.skip_videos:
        logger.info("=" * 60)
        logger.info("Step 7: Linking video files")
        logger.info("=" * 60)
        link_video_files(src_dir, output_dir, full_df, features, symlink=args.symlink)

    logger.info("=" * 60)
    logger.info(f"✅ Conversion complete! Dataset saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
