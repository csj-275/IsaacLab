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

数据格式（维度从源数据自动检测，以下仅为示例）:
  - action: 7维 (IK delta pose x/y/z/rx/ry/rz + gripper)
  - observation.state: N维 (取决于录制时的观测配置，常见 7)
  - observation.images.front: 1280x720 @ 30fps (如果存在)
  - observation.images.wrist: 1280x720 @ 30fps (如果存在)

归一化: 此脚本不修改原始数据。统计量 (mean/std/min/max) 保存到 meta/stats.json，
由训练时的 NormalizerProcessorStep / 推理时的 UnnormalizerProcessorStep 在线处理。
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
from datasets import Dataset, Features, Sequence, Value
from tqdm import tqdm

from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
from lerobot.datasets.utils import (
    create_empty_dataset_info,
    write_episodes,
    write_info,
    write_stats,
    write_tasks,
    DEFAULT_DATA_PATH,
    DEFAULT_EPISODES_PATH,
    DEFAULT_VIDEO_PATH,
)
from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


_STATE_NAMES = ["joint1.pos", "joint2.pos", "joint3.pos", "joint4.pos", "joint5.pos", "joint6.pos", "gripper.pos"]
"""Names for 7D joint position features (state or action)."""


def build_features(full_df: pd.DataFrame, video_keys: list[str] | None = None) -> dict:
    """构建 LeRobot features 字典，自动从数据检测维度."""
    sample = full_df.iloc[0]
    action_dim = len(sample["action"])
    state_dim = len(sample["observation.state"])

    # Action names: 6-DOF IK delta pose + gripper (from Mimic env)
    _ACTION_NAMES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
    if action_dim != len(_ACTION_NAMES):
        _ACTION_NAMES = None  # fallback if dim mismatch

    features = {
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "type": "ACTION",
            "names": _ACTION_NAMES,
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "type": "STATE",
            "names": _STATE_NAMES if (state_dim == len(_STATE_NAMES)) else None,
        },
    }

    # Camera intrinsics from grab_ik_rel_visuomotor_env_cfg (PIPER_D435_COLOR_INTRINSIC_1280X720)
    _CAMERA_INTRINSICS = {
        "width": 1280,
        "height": 720,
        "fx": 1211.038757324218,
        "fy": 908.279067993164,
        "ppx": 640.0,
        "ppy": 360.0,
        "model": "inverse_brown_conrady",
        "coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
        "stream": "color",
    }

    if video_keys is None:
        video_keys = ["observation.images.front", "observation.images.wrist"]

    for vk in video_keys:
        features[vk] = {
            "dtype": "video",
            "shape": (720, 1280, 3),
            "type": "VISUAL",
            "names": ["height", "width", "channel"],
            "info": {
                "camera_intrinsics": dict(_CAMERA_INTRINSICS),
            },
        }

    return features


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


def save_data_parquet_files(full_df: pd.DataFrame, output_dir: Path, features: dict,
                           use_state_as_action: bool = False) -> None:
    """用纯 pyarrow 写 parquet（不带 HuggingFace 元数据），FixedSizeList Schema。

    每 episode 一个 file-{ep_idx:03d}.parquet，放在 chunk-000 下。
    """
    data_pattern = DEFAULT_DATA_PATH

    if use_state_as_action:
        # Split 14D observation.state = [actual_joints(7) | target_joints(7)] into:
        #   action = target_joints (last 7D)
        #   observation.state = actual_joints (first 7D)
        full_df = full_df.copy()
        full_df["action"] = full_df["observation.state"].apply(lambda x: x[7:])   # targets
        full_df["observation.state"] = full_df["observation.state"].apply(lambda x: x[:7])  # actuals
        # Update feature shapes
        features["action"]["shape"] = (7,)
        features["observation.state"]["shape"] = (7,)
        features["action"]["names"] = _STATE_NAMES  # joint position names
        logger.info("Split 14D state → action(7D targets) + state(7D actuals)")

    action_len = features["action"]["shape"][0]
    state_len = features["observation.state"]["shape"][0]

    # 构建完全匹配 HF datasets.Sequence 的 FixedSizeList Arrow 类型
    action_type = pa.list_(pa.field("item", pa.float32(), nullable=False), list_size=action_len)
    state_type = pa.list_(pa.field("item", pa.float32(), nullable=False), list_size=state_len)

    for ep_idx, group in tqdm(full_df.groupby("episode_index"), desc="Saving data parquet"):
        ep_idx = int(ep_idx)
        group = group.sort_values("frame_index")

        action_arrays = [pa.array(row, type=pa.float32()) for row in group["action"].values]
        state_arrays = [pa.array(row, type=pa.float32()) for row in group["observation.state"].values]

        table = pa.table({
            "action": pa.array(action_arrays, type=action_type),
            "observation.state": pa.array(state_arrays, type=state_type),
            "timestamp": pa.array(group["timestamp"].values, type=pa.float32()),
            "frame_index": pa.array(group["frame_index"].values, type=pa.int64()),
            "episode_index": pa.array(group["episode_index"].values, type=pa.int64()),
            "index": pa.array(group["index"].values, type=pa.int64()),
            "task_index": pa.array(group["task_index"].values, type=pa.int64()),
        })

        fpath = output_dir / data_pattern.format(chunk_index=0, file_index=ep_idx)
        Path(fpath).parent.mkdir(parents=True, exist_ok=True)
        # 显式指定 store_schema=False 避免写入 HuggingFace 元数据
        pq.write_table(table, fpath, store_schema=True)

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
    parser.add_argument("--use-state-as-action", action="store_true",
                        help="Use observation.state as action (joint-position training)")
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

    # 2. 构建 features（自动检测 action/state 维度）
    video_keys = ["observation.images.front", "observation.images.wrist"]
    use_videos = not args.skip_videos
    if args.skip_videos:
        video_keys = []
    features = build_features(full_df, video_keys=video_keys)
    if args.use_state_as_action:
        # Swap action names to joint position names
        state_names = features["observation.state"]["names"]
        features["action"]["names"] = state_names

    # 3. 创建 info.json
    logger.info("=" * 60)
    logger.info("Step 2: Creating meta/info.json")
    logger.info("=" * 60)
    info = create_empty_dataset_info(
        codebase_version=CODEBASE_VERSION,
        fps=30,
        features=features,
        use_videos=use_videos,
        robot_type="piper",
        chunks_size=1000,
        data_files_size_in_mb=100,
        video_files_size_in_mb=200,
    )
    # 更新 total_episodes, total_frames, total_tasks
    info["total_episodes"] = full_df["episode_index"].nunique()
    info["total_frames"] = len(full_df)
    info["total_tasks"] = 1
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
    save_data_parquet_files(full_df, output_dir, features, use_state_as_action=args.use_state_as_action)

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
