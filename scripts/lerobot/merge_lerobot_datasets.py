#!/usr/bin/env python3
"""Merge multiple LeRobot-format datasets into one.

Usage:
    python merge_lerobot_datasets.py \
        --input datasets/A datasets/B datasets/C \
        --output datasets/merged

    python merge_lerobot_datasets.py \
        --input datasets/F-SIM-PIPER-GRAB-0721-N30-K-V1 \
               datasets/D-SIM-PIPER-GRAB-0721-N100-K-V1-L1 \
        --output datasets/H-SIM-PIPER-GRAB-0724-L1

The script:
  1. Copies data parquet files with reindexed episode_index and global index
  2. Updates episode metadata (data/video file paths, frame indices, timestamps)
  3. Copies video files with matching indices
  4. Updates info.json and recomputes stats.json
  5. Handles datasets with or without videos
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def read_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def write_json(path: str, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def collect_video_keys(ds_path: Path) -> list[str]:
    """Return sorted list of video keys (e.g. ['observation.images.front', 'observation.images.wrist'])."""
    vdir = ds_path / "videos"
    if not vdir.exists():
        return []
    return sorted([d.name for d in vdir.iterdir() if d.is_dir()])


def get_video_files(ds_path: Path, video_key: str) -> list[Path]:
    """Return sorted list of video files for a given key (e.g. chunk-000/file-XXX.mp4)."""
    chunk_dir = ds_path / "videos" / video_key / "chunk-000"
    if not chunk_dir.exists():
        return []
    return sorted(chunk_dir.glob("file-*.mp4"))


def merge_datasets(inputs: list[str], output: str):
    ds_paths = [Path(p) for p in inputs]
    out = Path(output)

    # Validate all input datasets exist
    for i, ds_path in enumerate(ds_paths):
        assert ds_path.exists(), f"Input dataset {i+1} not found: {ds_path}"

    # Load info for all datasets
    infos = []
    for i, ds_path in enumerate(ds_paths):
        infos.append(read_json(str(ds_path / "meta" / "info.json")))

    # Validate compatibility — all datasets must match the first
    base_info = infos[0]
    for i, info in enumerate(infos[1:], start=2):
        assert info["robot_type"] == base_info["robot_type"], f"Robot type mismatch in dataset {i}"
        assert info["features"] == base_info["features"], f"Feature definitions mismatch in dataset {i}"
        assert info["fps"] == base_info["fps"], f"FPS mismatch in dataset {i}"

    # Compute accumulated offsets
    eps_list = [info["total_episodes"] for info in infos]
    frames_list = [info["total_frames"] for info in infos]
    total_ep = sum(eps_list)
    total_frames = sum(frames_list)

    for i, (ds_path, ep, nframes) in enumerate(zip(ds_paths, eps_list, frames_list), start=1):
        print(f"Dataset {i}: {ds_path.name} - {ep} episodes, {nframes} frames")
    print(f"Output   : {out.name} - {total_ep} episodes, {total_frames} frames")

    # Detect video keys across all datasets
    all_video_keys = sorted(set().union(*[collect_video_keys(ds) for ds in ds_paths]))
    print(f"Video keys: {all_video_keys}")

    # ── Create output directories ───────────────────────────────────────
    out_data_dir = out / "data" / "chunk-000"
    out_ep_dir = out / "meta" / "episodes" / "chunk-000"
    out_data_dir.mkdir(parents=True, exist_ok=True)
    out_ep_dir.mkdir(parents=True, exist_ok=True)

    # ── Process datasets sequentially ────────────────────────────────────
    # Track accumulated frame index offset and episode index offset
    ds_configs = []
    ep_offset = 0
    frame_offset = 0
    for ds_path, ep_count, frame_count in zip(ds_paths, eps_list, frames_list):
        ds_configs.append({"path": ds_path, "ep_offset": ep_offset, "frame_offset": frame_offset})
        ep_offset += ep_count
        frame_offset += frame_count

    # Accumulators for episode metadata rows (one row per episode across all datasets)
    all_ep_meta_rows = []
    # Track video file index per video key (sequential across datasets)
    video_file_counters = {k: 0 for k in all_video_keys}
    total_data_files = 0

    for ds_cfg in ds_configs:
        ds_path = ds_cfg["path"]
        ep_offset = ds_cfg["ep_offset"]
        frame_offset = ds_cfg["frame_offset"]

        data_dir = ds_path / "data" / "chunk-000"
        data_files = sorted(data_dir.glob("file-*.parquet"))
        print(f"\nProcessing {ds_path.name}: {len(data_files)} episodes")

        # ── Read episode metadata (single parquet with all episodes) ────
        ep_meta_dir = ds_path / "meta" / "episodes" / "chunk-000"
        ep_meta_file = next(ep_meta_dir.glob("file-*.parquet"), None)
        if ep_meta_file is None:
            raise FileNotFoundError(f"No episode metadata found in {ep_meta_dir}")
        ep_meta_df = pd.read_parquet(ep_meta_file)
        # Convert to list of dicts for row-by-row updates
        ep_rows = ep_meta_df.to_dict("records")
        print(f"  Episode meta: {len(ep_rows)} rows in 1 file")

        # ── Process each episode ────────────────────────────────────────
        for pf, ep_row in zip(data_files, ep_rows):
            # Data: copy with reindexed indices
            df = pd.read_parquet(pf)
            df["episode_index"] = df["episode_index"] + ep_offset
            df["index"] = df["index"] + frame_offset
            out_data_path = out_data_dir / f"file-{total_data_files:03d}.parquet"
            df.to_parquet(out_data_path, index=False)

            # Episode meta: update indices and paths
            ep_row["episode_index"] = ep_row["episode_index"] + ep_offset
            ep_row["data/chunk_index"] = 0
            ep_row["data/file_index"] = total_data_files
            ep_row["dataset_from_index"] = df["index"].iloc[0]   # already offset
            ep_row["dataset_to_index"] = df["index"].iloc[-1]    # already offset
            ep_row["length"] = len(df)

            # Update video paths
            for vk in all_video_keys:
                ck = f"videos/{vk}/chunk_index"
                fk = f"videos/{vk}/file_index"
                src_video_dir = ds_path / "videos" / vk / "chunk-000"
                orig_video_idx = ep_row.get(fk, -1)

                if src_video_dir.exists() and orig_video_idx is not None and orig_video_idx >= 0:
                    new_video_idx = video_file_counters[vk]
                    ep_row[ck] = 0
                    ep_row[fk] = new_video_idx
                    video_file_counters[vk] += 1
                else:
                    ep_row[ck] = None
                    ep_row[fk] = None

            all_ep_meta_rows.append(ep_row)
            total_data_files += 1

    # ── Write merged episode metadata (single parquet with all episodes) ──
    merged_ep_meta = pd.DataFrame(all_ep_meta_rows)
    merged_ep_meta.to_parquet(out_ep_dir / "file-000.parquet", index=False)
    print(f"\nWrote {total_data_files} data files + 1 episode meta file")

    # ── Copy videos with reindexed names ─────────────────────────────────
    for vk in all_video_keys:
        out_video_dir = out / "videos" / vk / "chunk-000"
        out_video_dir.mkdir(parents=True, exist_ok=True)

        video_idx = 0
        for ds_path in ds_paths:
            src_video_dir = ds_path / "videos" / vk / "chunk-000"
            if not src_video_dir.exists():
                continue
            video_files = sorted(src_video_dir.glob("file-*.mp4"))
            for vf in video_files:
                dst = out_video_dir / f"file-{video_idx:03d}.mp4"
                if not dst.exists():
                    shutil.copy2(vf, dst)
                video_idx += 1
        print(f"  {vk}: {video_idx} video files copied")

    # ── Update info.json ─────────────────────────────────────────────────
    info_out = base_info.copy()
    info_out["total_episodes"] = total_ep
    info_out["total_frames"] = total_frames
    write_json(str(out / "meta" / "info.json"), info_out)

    # ── Recompute stats.json ────────────────────────────────────────────
    print("Recomputing stats...")
    stats = recompute_stats(out)
    write_json(str(out / "meta" / "stats.json"), stats)

    # ── Copy tasks.parquet ──────────────────────────────────────────────
    # Merge them if they differ; otherwise copy from the first
    tasks_all = [pd.read_parquet(ds / "meta" / "tasks.parquet") for ds in ds_paths]
    if all(t.equals(tasks_all[0]) for t in tasks_all[1:]):
        shutil.copy(ds_paths[0] / "meta" / "tasks.parquet", out / "meta" / "tasks.parquet")
    else:
        # Merge with reindex across all datasets
        merged_tasks = []
        task_offset = 0
        for t in tasks_all:
            t = t.copy()
            if task_offset > 0:
                t["task_index"] = t["task_index"] + task_offset
            merged_tasks.append(t)
            task_offset = t["task_index"].max() + 1
        pd.concat(merged_tasks, ignore_index=True).to_parquet(out / "meta" / "tasks.parquet", index=False)
        print("Merged tasks.parquet (different task definitions)")

    # ── Verify ───────────────────────────────────────────────────────────
    verify(out, total_ep, total_frames, all_video_keys)

    print(f"\nDone! Merged dataset at: {out}")
    print(f"  Episodes: {total_ep}")
    print(f"  Frames:   {total_frames}")


def verify(out: Path, expected_ep: int, expected_frames: int, video_keys: list[str]):
    """Sanity check the merged dataset."""
    info = read_json(str(out / "meta" / "info.json"))
    assert info["total_episodes"] == expected_ep
    assert info["total_frames"] == expected_frames

    data_files = sorted((out / "data" / "chunk-000").glob("file-*.parquet"))
    ep_meta_file = next((out / "meta" / "episodes" / "chunk-000").glob("file-*.parquet"), None)

    assert len(data_files) == expected_ep, f"Expected {expected_ep} data files, got {len(data_files)}"
    assert ep_meta_file is not None, "Missing episode metadata file"

    total_rows = 0
    episodes_seen = set()
    for pf in data_files:
        df = pd.read_parquet(pf)
        total_rows += len(df)
        episodes_seen.update(df["episode_index"].unique())

    assert total_rows == expected_frames, f"Expected {expected_frames} rows, got {total_rows}"
    assert len(episodes_seen) == expected_ep, f"Expected {expected_ep} unique episodes, got {len(episodes_seen)}"

    # Check video files match episode metadata
    ep_meta_df = pd.read_parquet(ep_meta_file)
    for vk in video_keys:
        video_dir = out / "videos" / vk / "chunk-000"
        if video_dir.exists():
            video_files = sorted(video_dir.glob("file-*.mp4"))
            fk = f"videos/{vk}/file_index"
            if fk in ep_meta_df.columns:
                valid = ep_meta_df[fk].dropna()
                video_eps = len(valid)
                max_vidx = int(valid.max()) if video_eps > 0 else -1
            else:
                video_eps, max_vidx = 0, -1
            print(f"  {vk}: {len(video_files)} video files, {video_eps} episodes reference them (max index {max_vidx})")
            if video_eps > 0:
                assert max_vidx + 1 == len(video_files), (
                    f"Video count mismatch: {len(video_files)} files but max episode index is {max_vidx}"
                )

    print("Verification passed!")


def recompute_stats(output_dir: Path) -> dict:
    """Recompute min, max, mean, std for action and observation.state from merged data."""
    data_dir = output_dir / "data" / "chunk-000"
    parquet_files = sorted(data_dir.glob("file-*.parquet"))

    all_data = {}
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        for col in df.columns:
            if col == "action" or col.startswith("observation"):
                vals = np.stack(df[col].to_numpy())  # type: ignore[arg-type]
                if col not in all_data:
                    all_data[col] = []
                all_data[col].append(vals)

    stats = {}
    for col, arrays in all_data.items():
        combined = np.concatenate(arrays, axis=0)
        stats[col] = {
            "min": combined.min(axis=0).tolist(),
            "max": combined.max(axis=0).tolist(),
            "mean": combined.mean(axis=0).tolist(),
            "std": combined.std(axis=0).tolist(),
        }

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge multiple LeRobot datasets into one")
    parser.add_argument("--input", nargs="+", required=True, help="Input dataset directories (one or more)")
    parser.add_argument("--output", required=True, help="Output merged dataset directory")
    args = parser.parse_args()

    merge_datasets(args.input, os.path.abspath(args.output))
