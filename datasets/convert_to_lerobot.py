#!/usr/bin/env python3
"""
Convert Isaac Lab HDF5 demonstration dataset to LeRobot v2.1 format.

Usage:
    # State-only conversion
    python convert_to_lerobot.py --input piper_dataset.hdf5 --output ./lerobot_data

    # With images (--enable_cameras was used during recording)
    python convert_to_lerobot.py --input visuo_dataset.hdf5 --output ./lerobot_data

    # Specify fps and custom state keys
    python convert_to_lerobot.py --input piper_dataset.hdf5 --output ./lerobot_data \\
        --fps 20 --state-keys joint_pos eef_pos eef_quat gripper_pos

LeRobot v2.1 output structure:
    <output_dir>/
    ├── data/chunk-000/
    │   └── episode_000000.parquet
    ├── videos/chunk-000/
    │   └── observation.images.<cam>/
    │       └── episode_000000.mp4
    └── meta/
        ├── info.json
        └── episodes.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# Detect optional image codec
# ---------------------------------------------------------------------------
try:
    import av

    HAS_PYAV = True
except ImportError:
    HAS_PYAV = False
    print("[warn] av (pyav) not installed — video/image conversion disabled. Install with: pip install av")

try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# All possible observation keys that the V1 visuomotor policy produces
ALL_OBS_KEYS = [
    "actions",  # last raw action
    "joint_pos",
    "joint_vel",
    "object",
    "eef_pos",
    "eef_quat",
    "gripper_pos",
    "object_1_positions",
    "object_1_orientations",
    "box_positions",
    "box_orientations",
    "mug_positions",
    "mug_orientations",
]

# Image-like observation keys (will be stored as videos, not in parquet)
IMAGE_OBS_KEYS = [
    "table_cam",
    "wrist_cam",
    "table_cam_depth",
    "wrist_cam_depth",
]


def _is_image_key(key: str, value: h5py.Dataset) -> bool:
    """Heuristic: image data has 3+ dims (T, H, W) or (T, H, W, C)."""
    return value.ndim >= 3


def _resolve_state_obs_keys(first_demo: h5py.Group, requested: list[str] | None) -> list[str]:
    """Determine which observation keys to include in observation.state.

    If *requested* is given, use exactly those (validating they exist in the
    first demo's ``obs/`` group).  Otherwise default to all non-image keys.
    """
    avail = set(first_demo["obs"].keys())
    if requested:
        missing = set(requested) - avail
        if missing:
            print(f"[warn] requested state obs keys not in dataset: {missing}")
        return [k for k in requested if k in avail]
    # Default: all non-image keys
    return sorted(k for k in avail if k in first_demo["obs"] and not _is_image_key(k, first_demo["obs"][k]))


def _collect_image_obs_keys(first_demo: h5py.Group) -> list[str]:
    """Return image-like observation keys present in the dataset."""
    return sorted(
        k
        for k in first_demo["obs"].keys()
        if _is_image_key(k, first_demo["obs"][k]) and first_demo["obs"][k].ndim >= 3
    )


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------
def _encode_mp4_pyav(frames: np.ndarray, path: Path, fps: int) -> None:
    """Encode (T, H, W, 3) uint8 frames as an MP4 file using PyAV."""
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = frames.shape[2]
    stream.height = frames.shape[1]
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "23", "preset": "fast"}

    for frame in frames:
        av_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        for packet in stream.encode(av_frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()


def _encode_mp4_cv2(frames: np.ndarray, path: Path, fps: int) -> None:
    """Encode (T, H, W, 3) uint8 frames as an MP4 file using OpenCV."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (frames.shape[2], frames.shape[1]))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def encode_video(frames: np.ndarray, path: Path, fps: int) -> None:
    """Encode (T, H, W, 3) uint8 frames as MP4, preferring PyAV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if HAS_PYAV:
        _encode_mp4_pyav(frames, path, fps)
    elif HAS_CV2:
        _encode_mp4_cv2(frames, path, fps)
    else:
        raise RuntimeError("No video encoder available. Install av or opencv-python.")


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------
def convert_hdf5_to_lerobot(
    input_path: str | Path,
    output_dir: str | Path,
    fps: int = 20,
    state_obs_keys: list[str] | None = None,
    action_source: str = "actions",
    skip_videos: bool = False,
) -> None:
    """Convert an Isaac Lab HDF5 dataset to LeRobot v2.1 format.

    Args:
        input_path:   Path to the HDF5 file produced by ``record_demos.py``.
        output_dir:   Where to write the LeRobot dataset tree.
        fps:          Frames-per-second metadata (policy frequency).
        state_obs_keys: Which ``obs/`` keys to concatenate into ``observation.state``.
                        Defaults to all non-image observation keys.
        action_source: Which dataset to use for actions (``"actions"`` or
                       ``"processed_actions"``).
        skip_videos:  If True, do not write MP4 video files even when images are present.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    # -- Open HDF5 -----------------------------------------------------------
    f = h5py.File(input_path, "r")
    data_group = f["data"]

    env_args = json.loads(data_group.attrs.get("env_args", "{}"))
    total_steps = int(data_group.attrs.get("total", 0))

    demo_keys = sorted(
        (k for k in data_group.keys() if k.startswith("demo_")),
        key=lambda x: int(x.split("_")[1]),
    )
    print(f"Found {len(demo_keys)} episodes, {total_steps} total steps")

    if not demo_keys:
        print("No demos found — aborting.")
        f.close()
        return

    # -- Probe first demo for observation structure --------------------------
    first = data_group[demo_keys[0]]
    has_obs = "obs" in first
    has_images = False

    if has_obs:
        state_keys = _resolve_state_obs_keys(first, state_obs_keys)
        image_keys = _collect_image_obs_keys(first)
        has_images = len(image_keys) > 0
    else:
        state_keys = []
        image_keys = []

    # Validate action source
    if action_source not in first:
        raise KeyError(
            f"Action source '{action_source}' not found in demo. Available: {list(first.keys())}"
        )

    # Compute state dimension from first demo
    state_dim = 0
    if state_keys:
        for k in state_keys:
            state_dim += first["obs"][k].shape[-1]
    else:
        # Fallback: use joint_position from states if no obs group
        if "states" in first:
            sj = first["states"]["articulation"]["robot"]["joint_position"]
            state_dim = sj.shape[-1]
            state_keys = []  # signal to extract from states/ instead

    action_dim = first[action_source].shape[-1]

    print(f"  State dimension: {state_dim} (keys: {state_keys or 'states/joint_position'})")
    print(f"  Action dimension: {action_dim} (source: {action_source})")
    if has_images:
        print(f"  Image observations: {image_keys}")
    if skip_videos:
        print(f"  Video encoding: SKIPPED")

    # -- Create directory structure ------------------------------------------
    (output_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (output_dir / "meta").mkdir(parents=True, exist_ok=True)

    video_base = output_dir / "videos" / "chunk-000"

    # -- Convert each episode -------------------------------------------------
    total_frames = 0
    episodes_meta = []

    for ep_idx, demo_key in enumerate(demo_keys):
        demo = data_group[demo_key]
        num_samples = int(demo.attrs.get("num_samples", 0))
        success = bool(demo.attrs.get("success", False))

        # ---- Build state vector --------------------------------------------
        if state_keys:
            state_parts = []
            for k in state_keys:
                arr = demo["obs"][k][:]  # (T, D_k)
                state_parts.append(arr)
            state = np.concatenate(state_parts, axis=-1).astype(np.float32)
        else:
            # Fallback: use joint_position from states
            state = demo["states"]["articulation"]["robot"]["joint_position"][:].astype(np.float32)

        # ---- Action vector --------------------------------------------------
        action = demo[action_source][:].astype(np.float32)

        # Sanity-check lengths
        actual_len = min(state.shape[0], action.shape[0])
        if actual_len != num_samples:
            print(f"  [{demo_key}] num_samples={num_samples} but data shape={actual_len}; trimming.")
            num_samples = actual_len

        state = state[:num_samples]
        action = action[:num_samples]

        # ---- Write parquet -------------------------------------------------
        timestamp = np.arange(num_samples, dtype=np.float32) / fps
        episode_index = np.full(num_samples, ep_idx, dtype=np.int64)
        frame_index = np.arange(num_samples, dtype=np.int64)
        index = np.arange(total_frames, total_frames + num_samples, dtype=np.int64)
        done = np.zeros(num_samples, dtype=bool)
        done[-1] = True

        columns = {
            "observation.state": state,
            "action": action,
            "timestamp": timestamp,
            "episode_index": episode_index,
            "frame_index": frame_index,
            "index": index,
            "next.done": done,
        }

        # Add task_index (always 0 for single-task dataset)
        columns["task_index"] = np.zeros(num_samples, dtype=np.int64)

        # ---- Handle images → MP4 videos ------------------------------------
        if has_images and not skip_videos:
            for cam_key in image_keys:
                frames = demo["obs"][cam_key][:num_samples]
                # Depth images are (T, H, W), RGB are (T, H, W, 3)
                if frames.ndim == 3:
                    # Normalized depth → scale to uint8 for video
                    frames = frames.astype(np.float32)
                    f_min, f_max = frames.min(), frames.max()
                    if f_max - f_min > 0:
                        frames = (frames - f_min) / (f_max - f_min) * 255.0
                    frames = frames.astype(np.uint8)
                    frames = np.stack([frames, frames, frames], axis=-1)  # grayscale→RGB
                elif frames.ndim == 4 and frames.shape[-1] != 3:
                    frames = frames[..., :3]  # RGBA → RGB

                video_subdir = video_base / f"observation.images.{cam_key}"
                video_subdir.mkdir(parents=True, exist_ok=True)
                video_path = video_subdir / f"episode_{ep_idx:06d}.mp4"

                print(f"  Encoding {cam_key} → {video_path} ({frames.shape[0]} frames)")
                encode_video(frames, video_path, fps)

        # Write parquet file (use pyarrow directly for multi-dim arrays)
        import pyarrow as pa
        import pyarrow.parquet as pq

        arrays = []
        for col_name, col_data in columns.items():
            if col_data.ndim > 1:
                # Multi-dim → list of fixed-size lists
                pa_arr = pa.FixedSizeListArray.from_arrays(
                    pa.array(col_data.ravel(), type=pa.from_numpy_dtype(col_data.dtype)),
                    list_size=col_data.shape[-1],
                )
            elif col_data.dtype == bool:
                pa_arr = pa.array(col_data, type=pa.bool_())
            else:
                pa_arr = pa.array(col_data)
            arrays.append(pa_arr)

        table = pa.table(dict(zip(columns.keys(), arrays)))
        parquet_path = output_dir / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
        pq.write_table(table, parquet_path)

        # ---- Metadata -------------------------------------------------------
        episodes_meta.append(
            {
                "episode_index": ep_idx,
                "tasks": [],
                "length": num_samples,
                "success": success,
            }
        )

        total_frames += num_samples
        print(f"  [{demo_key}] {num_samples} steps, success={success}")

    f.close()

    # -- Write meta/info.json ------------------------------------------------
    features = {
        "observation.state": {"dtype": "float32", "shape": [state_dim]},
        "action": {"dtype": "float32", "shape": [action_dim]},
        "timestamp": {"dtype": "float32", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "index": {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
        "next.done": {"dtype": "bool", "shape": [1]},
    }

    if has_images and not skip_videos:
        for cam_key in image_keys:
            features[f"observation.images.{cam_key}"] = {"dtype": "video", "shape": [480, 640, 3]}

    info = {
        "codebase_version": "v2.1",
        "robot_type": "piper",
        "fps": fps,
        "total_episodes": len(demo_keys),
        "total_frames": total_frames,
        "total_tasks": 1,
        "features": features,
    }

    with open(output_dir / "meta" / "info.json", "w") as fp:
        json.dump(info, fp, indent=2)

    # -- Write meta/episodes.jsonl -------------------------------------------
    with open(output_dir / "meta" / "episodes.jsonl", "w") as fp:
        for ep in episodes_meta:
            fp.write(json.dumps(ep) + "\n")

    # -- Write meta/tasks.jsonl ----------------------------------------------
    tasks = [{"task_index": 0, "task": "pick cube and mug into box"}]
    with open(output_dir / "meta" / "tasks.jsonl", "w") as fp:
        for t in tasks:
            fp.write(json.dumps(t) + "\n")

    print(f"\nDone. LeRobot dataset written to {output_dir.resolve()}")
    print(f"  Episodes: {len(demo_keys)}")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps}")
    print(f"  State dim: {state_dim}")
    print(f"  Action dim: {action_dim}")
    if has_images and not skip_videos:
        print(f"  Cameras: {image_keys}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Convert Isaac Lab HDF5 demo dataset to LeRobot v2.1 format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", "-i", required=True, help="Path to input .hdf5 file")
    parser.add_argument("--output", "-o", required=True, help="Output directory for LeRobot dataset")
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Policy frequency in Hz (default: 20, matches decimation=5 at dt=0.01)",
    )
    parser.add_argument(
        "--state-keys",
        nargs="*",
        default=None,
        help="obs/ keys to include in observation.state. Default: all non-image obs keys. "
        f"Available: {', '.join(ALL_OBS_KEYS)}",
    )
    parser.add_argument(
        "--action-source",
        default="actions",
        choices=["actions", "processed_actions"],
        help="Which dataset to use as action (default: actions = delta-pose 7-dof)",
    )
    parser.add_argument(
        "--skip-videos",
        action="store_true",
        help="Skip video encoding even if images are present",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Force state-only mode, ignore image observations",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: input file not found: {args.input}")
        sys.exit(1)

    convert_hdf5_to_lerobot(
        input_path=args.input,
        output_dir=args.output,
        fps=args.fps,
        state_obs_keys=args.state_keys,
        action_source=args.action_source,
        skip_videos=args.skip_videos or args.no_images,
    )


if __name__ == "__main__":
    main()
