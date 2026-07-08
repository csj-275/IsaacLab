# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LeRobot dataset file handler for storing episode data in parquet + MP4 format.

Matches the reference format used by SIM-PIPER-GRAB-0604-N25-V1:
    - codebase_version: v3.0
    - Naming: file-{idx:03d}.parquet / file-{idx:03d}.mp4
    - Parquet columns: action, observation.state, timestamp, frame_index,
      episode_index, index, task_index (no next.done)
    - Video paths: videos/observation.images.<cam>/chunk-000/file-<idx>.mp4
    - Meta: info.json, stats.json, tasks.parquet, episodes jsonl+parquet,
      recording_meta.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from .dataset_file_handler_base import DatasetFileHandlerBase

if TYPE_CHECKING:
    from .episode_data import EpisodeData

# ---------------------------------------------------------------------------
# Optional video encoder detection (ffmpeg preferred, OpenCV as fallback)
# ---------------------------------------------------------------------------
import shutil

_HAS_FFMPEG = shutil.which("ffmpeg") is not None

try:
    import cv2 as _cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

if not _HAS_FFMPEG and not HAS_CV2:
    raise RuntimeError("No video encoder available. Install ffmpeg or opencv-python.")


# ---------------------------------------------------------------------------
# Camera name mapping: Isaac Lab obs keys → LeRobot feature names
# ---------------------------------------------------------------------------
_OBS_KEY_TO_LEROBOT_CAM: dict[str, str] = {
    "table_cam": "front",
    "table_cam_depth": "front",
    "wrist_cam": "wrist",
    "wrist_cam_depth": "wrist",
}


def _lerobot_cam_name(obs_key: str) -> str:
    """Map an Isaac Lab camera obs key to a LeRobot camera name."""
    if obs_key in _OBS_KEY_TO_LEROBOT_CAM:
        return _OBS_KEY_TO_LEROBOT_CAM[obs_key]
    # Fallback: strip common suffixes and use as-is
    for suffix in ("_cam", "_depth", "_rgb", "_image"):
        obs_key = obs_key.replace(suffix, "")
    return obs_key


def _is_depth_key(obs_key: str) -> bool:
    """Check if the obs key refers to a depth image."""
    return "depth" in obs_key.lower()


# ---------------------------------------------------------------------------
# Default camera intrinsics (matches reference dataset)
# ---------------------------------------------------------------------------
DEFAULT_CAMERA_INTRINSICS = {
    "width": 1280,
    "height": 720,
    "fx": 1211.04,
    "fy": 908.28,
    "ppx": 640.0,
    "ppy": 360.0,
    "model": "inverse_brown_conrady",
    "coeffs": [0.0] * 12,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_image_value(value: torch.Tensor) -> bool:
    """Heuristic: image data has 3+ dims (T, H, W, ...)."""
    return value.ndim >= 3


def _encode_video_ffmpeg(frames: np.ndarray, path: Path, fps: int) -> None:
    """Encode (T, H, W, 3) uint8 frames as MP4 using ffmpeg.

    Frames are piped as raw RGB24 bytes via stdin, matching the approach
    used in ``hdf5_to_mp4.py``.
    """
    import subprocess

    T, H, W, C = frames.shape
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{W}x{H}", "-pix_fmt", "rgb24", "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    proc.communicate(frames.tobytes())
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {path}")


def _encode_video_cv2(frames: np.ndarray, path: Path, fps: int) -> None:
    """Fallback: encode with OpenCV."""
    fourcc = _cv2.VideoWriter_fourcc(*"mp4v")
    writer = _cv2.VideoWriter(str(path), fourcc, fps, (frames.shape[2], frames.shape[1]))
    for frame in frames:
        writer.write(_cv2.cvtColor(frame, _cv2.COLOR_RGB2BGR))
    writer.release()


def _encode_video(frames: np.ndarray, path: Path, fps: int) -> None:
    """Encode frames as MP4, preferring ffmpeg."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if _HAS_FFMPEG:
        _encode_video_ffmpeg(frames, path, fps)
    elif HAS_CV2:
        _encode_video_cv2(frames, path, fps)
    else:
        raise RuntimeError("No video encoder available. Install ffmpeg or opencv-python.")


def _normalize_depth_for_video(depth: np.ndarray) -> np.ndarray:
    """Normalize (T, H, W) depth to uint8 3-channel for video encoding."""
    depth = depth.astype(np.float32)
    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min > 0:
        depth = (depth - d_min) / (d_max - d_min) * 255.0
    depth = depth.astype(np.uint8)
    return np.stack([depth, depth, depth], axis=-1)


def _prepare_image_frames(frames: np.ndarray) -> np.ndarray:
    """Convert raw image tensor to uint8 RGB (T, H, W, 3) for MP4 encoding.

    Handles: RGB, RGBA (→RGB), depth (T,H,W or T,H,W,1 → normalized RGB).
    """
    if frames.ndim == 3:
        return _normalize_depth_for_video(frames)
    if frames.ndim == 4 and frames.shape[-1] == 1:
        return _normalize_depth_for_video(frames[..., 0])
    if frames.ndim == 4 and frames.shape[-1] == 4:
        return frames[..., :3].astype(np.uint8)
    if frames.dtype != np.uint8:
        frames = frames.astype(np.uint8)
    return frames


def _generate_state_names(state_keys: list[str], dims: dict[str, int]) -> list[str]:
    """Generate human-readable names for observation.state dimensions.

    Args:
        state_keys: Ordered list of observation key names (e.g. ['joint_pos', 'eef_pos', ...]).
        dims: Mapping from key to number of dimensions.

    Returns:
        List of per-dimension names, e.g. ['joint1.pos', ..., 'gripper.pos', 'eef.x', ...].
    """
    names: list[str] = []
    for key in state_keys:
        d = dims.get(key, 1)
        if "joint_pos" in key or key == "joint_pos":
            if d == 7:
                names += [
                    "joint1.pos", "joint2.pos", "joint3.pos",
                    "joint4.pos", "joint5.pos", "joint6.pos",
                    "gripper.pos",
                ]
            else:
                for i in range(d):
                    names.append(f"{key}.dim_{i}")
        elif "joint_vel" in key or key == "joint_vel":
            if d == 7:
                names += [
                    "joint1.vel", "joint2.vel", "joint3.vel",
                    "joint4.vel", "joint5.vel", "joint6.vel",
                    "gripper.vel",
                ]
            else:
                for i in range(d):
                    names.append(f"{key}.dim_{i}")
        elif "eef_pos" in key or key == "eef_pos":
            names += ["eef.x", "eef.y", "eef.z"]
        elif "eef_quat" in key or key == "eef_quat":
            names += ["eef.qw", "eef.qx", "eef.qy", "eef.qz"]
        elif "gripper_pos" in key or key == "gripper_pos":
            if d == 1:
                names.append("gripper.pos")
            else:
                names.append(key)
        elif "object" in key:
            for i in range(d):
                names.append(f"{key}.dim_{i}")
        else:
            for i in range(d):
                names.append(f"{key}.dim_{i}")
    return names


def _generate_action_names(action_dim: int) -> list[str]:
    """Generate human-readable names for action dimensions."""
    if action_dim == 7:
        return [
            "joint1.pos", "joint2.pos", "joint3.pos",
            "joint4.pos", "joint5.pos", "joint6.pos",
            "gripper.pos",
        ]
    elif action_dim == 8:
        return ["dx", "dy", "dz", "dqx", "dqy", "dqz", "dqw", "gripper"]
    else:
        return [f"action.dim_{i}" for i in range(action_dim)]


def _find_joint_pos(states: dict) -> torch.Tensor | None:
    """Walk nested states dict to find joint_position."""
    if isinstance(states, torch.Tensor):
        return states
    for key in ["joint_position", "joint_positions", "joint_pos"]:
        if key in states:
            val = states[key]
            if isinstance(val, torch.Tensor):
                return val
    for key in ["articulation", "robot"]:
        if key in states and isinstance(states[key], dict):
            result = _find_joint_pos(states[key])
            if result is not None:
                return result
    return None


# ---------------------------------------------------------------------------
# LeRobot Dataset File Handler
# ---------------------------------------------------------------------------


class LeRobotDatasetFileHandler(DatasetFileHandlerBase):
    """LeRobot dataset file handler that saves directly in parquet + MP4 format.

    Matches the LeRobot v3.0 reference format. Each episode is written
    immediately to disk — no large intermediate HDF5 file is created.

    Output structure::

        <output_dir>/
        ├── data/chunk-000/
        │   └── file-000.parquet
        ├── videos/
        │   ├── observation.images.front/chunk-000/file-000.mp4
        │   ├── observation.images.wrist/chunk-000/file-000.mp4
        │   ├── observation.depths.front/chunk-000/file-000.mp4
        │   └── observation.depths.wrist/chunk-000/file-000.mp4
        └── meta/
            ├── info.json
            ├── stats.json
            ├── tasks.parquet
            ├── recording_meta.json
            └── episodes/chunk-000/
                ├── file-000.jsonl
                └── file-000.parquet
    """

    def __init__(self) -> None:
        """Initialise the LeRobot dataset file handler."""
        self._output_dir: Path | None = None
        self._env_name: str = ""
        self._env_args: dict = {}
        self._episode_count: int = 0
        self._total_frames: int = 0
        self._closed: bool = False
        self._fps: int = 30  # matching reference

        # Determined from the first episode
        self._state_dim: int | None = None
        self._state_key_names: list[str] = []
        self._state_key_dims: dict[str, int] | None = None
        self._action_dim: int | None = None

        # Camera-related
        self._image_keys: list[str] = []  # LeRobot feature keys, e.g. "observation.images.front"
        self._camera_intrinsics: dict = dict(DEFAULT_CAMERA_INTRINSICS)
        self._image_resolution: tuple[int, int] | None = None  # (H, W)

        # Episode metadata
        self._episodes_meta: list[dict] = []

        # Task description (overridable via setter)
        self._task_description: str = "pick the cube and place it into the box, then pick the bottle and place it into the box"

        # State key filter: if set, only these obs keys are included in observation.state.
        # None or empty → include all non-image obs keys.
        self._state_key_filter: list[str] | None = None

        # Action stats for stats.json (accumulated across all episodes)
        self._all_actions: list[np.ndarray] = []

    # ------------------------------------------------------------------
    # Configurable properties
    # ------------------------------------------------------------------

    @property
    def fps(self) -> int:
        """Frames per second for video encoding."""
        return self._fps

    @fps.setter
    def fps(self, value: int) -> None:
        self._fps = value

    @property
    def task_description(self) -> str:
        """Human-readable task description written to meta/tasks.parquet."""
        return self._task_description

    @task_description.setter
    def task_description(self, value: str) -> None:
        self._task_description = value

    @property
    def state_key_filter(self) -> list[str] | None:
        """If set, only these obs keys are included in observation.state. None = include all."""
        return self._state_key_filter

    @state_key_filter.setter
    def state_key_filter(self, value: list[str] | None) -> None:
        self._state_key_filter = value

    def set_camera_intrinsics(self, intrinsics: dict) -> None:
        """Override default camera intrinsics."""
        self._camera_intrinsics = dict(intrinsics)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def create(self, file_path: str, env_name: str | None = None) -> None:
        """Create a new LeRobot dataset directory structure.

        Args:
            file_path: Output directory path for the LeRobot dataset.
            env_name: Optional environment name for metadata.
        """
        if self._output_dir is not None:
            raise RuntimeError("LeRobot dataset file handler is already in use")

        self._output_dir = Path(file_path)
        self._env_name = env_name or ""

        # Create directory skeleton
        (self._output_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
        (self._output_dir / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)

    def open(self, file_path: str, mode: str = "r") -> None:
        """Open an existing dataset file (not supported for write-only handler)."""
        raise NotImplementedError(
            "Reading from LeRobot format is not yet implemented. "
            "Use HDF5DatasetFileHandler for reading input datasets."
        )

    def close(self) -> None:
        """Close the dataset and write all metadata files."""
        if self._closed:
            print("[LeRobotHandler] close() skipped: already closed")
            return
        if self._output_dir is None:
            print("[LeRobotHandler] close() skipped: _output_dir is None")
            self._closed = True
            return

        print(f"[LeRobotHandler] close() writing meta files... (episodes={self._episode_count}, total_frames={self._total_frames})")
        self._write_all_meta_files()
        self._closed = True
        print("[LeRobotHandler] close() done")

    def flush(self) -> None:
        """No-op — episodes are written immediately."""
        pass

    def __del__(self) -> None:
        """Destructor."""
        self.close()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    def add_env_args(self, env_args: dict) -> None:
        """Add environment arguments to the dataset metadata."""
        self._env_args.update(env_args)

    def set_env_name(self, env_name: str) -> None:
        """Set the environment name."""
        self._env_name = env_name

    def get_env_name(self) -> str | None:
        """Get the environment name."""
        return self._env_name

    def get_num_episodes(self) -> int:
        """Get number of episodes written so far."""
        return self._episode_count

    @property
    def demo_count(self) -> int:
        """Number of demos collected so far."""
        return self._episode_count

    # ------------------------------------------------------------------
    # Read operations (not supported)
    # ------------------------------------------------------------------

    def load_episode(self, episode_name: str, device: str | torch.device) -> EpisodeData | None:
        """Not supported — use HDF5Handler for input."""
        raise NotImplementedError("Use HDF5DatasetFileHandler for reading input datasets.")

    def get_episode_names(self) -> list[str]:
        """Get episode names."""
        return [f"episode_{i}" for i in range(self._episode_count)]

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def write_episode(self, episode: EpisodeData, demo_id: int | None = None) -> None:
        """Write a single episode as parquet + MP4 immediately.

        Args:
            episode: Episode data (``pre_export()`` already called).
            demo_id: Optional custom index. Defaults to internal counter.
        """
        self._raise_if_not_initialized()

        if episode.is_empty():
            return

        data = episode.data
        episode_idx = demo_id if demo_id is not None else self._episode_count
        success = bool(episode.success) if episode.success is not None else False

        # ---- Extract state & images from obs ----------------------------
        obs_dict = data.get("obs", {})
        state_parts: list[torch.Tensor] = []
        state_keys: list[str] = []
        state_dims: dict[str, int] = {}
        image_obs: dict[str, torch.Tensor] = {}

        if obs_dict:
            for key, value in obs_dict.items():
                if _is_image_value(value):
                    image_obs[key] = value
                elif self._state_key_filter and key not in self._state_key_filter:
                    continue  # skip keys not in the filter
                else:
                    state_parts.append(value)
                    state_keys.append(key)
                    state_dims[key] = value.shape[-1]

        if not state_parts:
            # Fallback: use joint_position from states
            states = data.get("states", {})
            if states:
                jp = _find_joint_pos(states)
                if jp is not None:
                    state_parts.append(jp)
                    state_keys.append("joint_pos")
                    state_dims["joint_pos"] = jp.shape[-1]

        if state_parts:
            state = torch.cat(state_parts, dim=-1).float()
        else:
            state = torch.zeros((0, 0), dtype=torch.float32)

        # Track state dimension / names from first episode
        if self._state_dim is None and state.shape[-1] > 0:
            self._state_dim = state.shape[-1]
            self._state_key_names = state_keys
            self._state_key_dims = state_dims

        # ---- Extract action ---------------------------------------------
        action_key = "processed_actions" if "processed_actions" in data else "actions"
        if action_key in data:
            action = data[action_key].float()
        else:
            action = torch.zeros((0, 0), dtype=torch.float32)

        if self._action_dim is None and action.shape[-1] > 0:
            self._action_dim = action.shape[-1]

        # ---- Sanity-check ------------------------------------------------
        num_samples = min(state.shape[0], action.shape[0]) if state.shape[0] > 0 else action.shape[0]
        if num_samples == 0:
            return

        state_np = state[:num_samples].cpu().numpy().astype(np.float32)
        action_np = action[:num_samples].cpu().numpy().astype(np.float32)

        # Accumulate actions for stats
        self._all_actions.append(action_np.copy())

        # ---- Write images as MP4 -----------------------------------------
        image_keys_written: list[str] = []  # LeRobot feature keys
        for obs_key, frames_tensor in image_obs.items():
            frames_np = frames_tensor[:num_samples].cpu().numpy()
            frames_np = _prepare_image_frames(frames_np)

            # Track image resolution
            if self._image_resolution is None and frames_np.ndim >= 3:
                self._image_resolution = (frames_np.shape[1], frames_np.shape[2])

            cam_name = _lerobot_cam_name(obs_key)
            if _is_depth_key(obs_key):
                feature_key = f"observation.depths.{cam_name}"
            else:
                feature_key = f"observation.images.{cam_name}"

            video_dir = self._output_dir / "videos" / feature_key / "chunk-000"
            video_dir.mkdir(parents=True, exist_ok=True)
            video_path = video_dir / f"file-{episode_idx:03d}.mp4"
            _encode_video(frames_np, video_path, self._fps)

            if feature_key not in image_keys_written:
                image_keys_written.append(feature_key)

        if image_keys_written and not self._image_keys:
            self._image_keys = sorted(image_keys_written)

        # ---- Build parquet columns (7 columns, NO next.done) -------------
        timestamp = np.arange(num_samples, dtype=np.float32) / self._fps

        columns = {
            "action": action_np,
            "observation.state": state_np,
            "timestamp": timestamp,
            "frame_index": np.arange(num_samples, dtype=np.int64),
            "episode_index": np.full(num_samples, episode_idx, dtype=np.int64),
            "index": np.arange(self._total_frames, self._total_frames + num_samples, dtype=np.int64),
            "task_index": np.zeros(num_samples, dtype=np.int64),
        }

        # ---- Write parquet -----------------------------------------------
        import pyarrow as pa
        import pyarrow.parquet as pq

        arrays = []
        for col_name, col_data in columns.items():
            if col_data.ndim > 1:
                inner_field = pa.field("item", pa.from_numpy_dtype(col_data.dtype), nullable=False)
                list_type = pa.list_(inner_field, list_size=col_data.shape[-1])
                pa_arr = pa.FixedSizeListArray.from_arrays(
                    pa.array(col_data.ravel(), type=pa.from_numpy_dtype(col_data.dtype)),
                    type=list_type,
                )
            else:
                pa_arr = pa.array(col_data)
            arrays.append(pa_arr)

        table = pa.table(dict(zip(columns.keys(), arrays)))
        parquet_path = self._output_dir / "data" / "chunk-000" / f"file-{episode_idx:03d}.parquet"
        pq.write_table(table, parquet_path)

        # ---- Track episode metadata --------------------------------------
        self._episodes_meta.append({
            "episode_index": episode_idx,
            "tasks": [],
            "length": num_samples,
            "success": success,
        })
        self._total_frames += num_samples
        self._episode_count += 1

    # ------------------------------------------------------------------
    # Metadata writing
    # ------------------------------------------------------------------

    def _raise_if_not_initialized(self) -> None:
        if self._output_dir is None:
            raise RuntimeError("LeRobot dataset file handler is not initialized. Call create() first.")

    def _write_all_meta_files(self) -> None:
        """Write all meta files matching the reference format."""
        if self._output_dir is None or self._episode_count == 0:
            print(f"[LeRobotHandler] _write_all_meta_files SKIPPED: _output_dir={self._output_dir}, _episode_count={self._episode_count}")
            return
        print(f"[LeRobotHandler] _write_all_meta_files writing info.json + stats.json + tasks.parquet...")

        meta_dir = self._output_dir / "meta"

        state_dim = self._state_dim or 0
        action_dim = self._action_dim or 0

        # Determine image resolution
        img_h, img_w = self._image_resolution or (720, 1280)

        # Generate state/action names
        if self._state_key_dims:
            state_names = _generate_state_names(self._state_key_names, self._state_key_dims)
        else:
            state_names = [f"state.dim_{i}" for i in range(state_dim)]
        action_names = _generate_action_names(action_dim)

        # ------------------------------------------------------------------
        # info.json
        # ------------------------------------------------------------------
        features = {
            "action": {
                "dtype": "float32",
                "names": action_names,
                "shape": [action_dim],
            },
            "observation.state": {
                "dtype": "float32",
                "names": state_names,
                "shape": [state_dim],
            },
        }

        for feat_key in self._image_keys:
            is_depth = "depths" in feat_key
            features[feat_key] = {
                "dtype": "video",
                "shape": [img_h, img_w, 1 if is_depth else 3],
                "names": ["height", "width", "channels"],
                "info": {
                    "camera_intrinsics": dict(self._camera_intrinsics),
                },
            }

        info = {
            "codebase_version": "v3.0",
            "fps": self._fps,
            "features": features,
            "total_episodes": self._episode_count,
            "total_frames": self._total_frames,
            "splits": {"train": f"0:{self._episode_count}"},
        }

        with open(meta_dir / "info.json", "w") as fp:
            json.dump(info, fp, indent=2)

        # ------------------------------------------------------------------
        # stats.json — action statistics across all episodes
        # ------------------------------------------------------------------
        all_actions = np.concatenate(self._all_actions, axis=0)  # (total_frames, action_dim)
        stats = {
            "action": {
                "min": all_actions.min(axis=0).tolist(),
                "max": all_actions.max(axis=0).tolist(),
                "mean": all_actions.mean(axis=0).tolist(),
                "std": all_actions.std(axis=0).tolist(),
                "q01": np.percentile(all_actions, 1, axis=0).tolist(),
                "q99": np.percentile(all_actions, 99, axis=0).tolist(),
            }
        }

        with open(meta_dir / "stats.json", "w") as fp:
            json.dump(stats, fp, indent=2)

        # ------------------------------------------------------------------
        # tasks.parquet
        # ------------------------------------------------------------------
        import pyarrow as pa
        import pyarrow.parquet as pq

        tasks_table = pa.table({
            "task_index": pa.array([0], type=pa.int64()),
            "task": pa.array([self._task_description]),
        })
        pq.write_table(tasks_table, meta_dir / "tasks.parquet")

        # ------------------------------------------------------------------
        # episodes JSONL
        # ------------------------------------------------------------------
        episodes_meta_dir = meta_dir / "episodes" / "chunk-000"
        episodes_meta_dir.mkdir(parents=True, exist_ok=True)

        with open(episodes_meta_dir / "file-000.jsonl", "w") as fp:
            for ep in self._episodes_meta:
                fp.write(json.dumps(ep) + "\n")

        # ------------------------------------------------------------------
        # episodes parquet
        # ------------------------------------------------------------------
        num_eps = self._episode_count
        ep_idx_arr = [ep["episode_index"] for ep in self._episodes_meta]
        ep_lengths = [ep["length"] for ep in self._episodes_meta]

        # Compute dataset_to_index (cumulative sum of lengths)
        dataset_from = []
        dataset_to = []
        running = 0
        for length in ep_lengths:
            dataset_from.append(running)
            running += length
            dataset_to.append(running)

        # Build timestamp ranges (each episode starts at 0.0, ends at length/fps)
        from_ts = [0.0] * num_eps
        to_ts = [round(length / self._fps, 4) for length in ep_lengths]

        ep_parquet_columns = {
            "episode_index": pa.array(ep_idx_arr, type=pa.int64()),
            "tasks": pa.array([[] for _ in range(num_eps)], type=pa.list_(pa.int64())),
            "length": pa.array(ep_lengths, type=pa.int64()),
            "data/chunk_index": pa.array([0] * num_eps, type=pa.int64()),
            "data/file_index": pa.array(list(range(num_eps)), type=pa.int64()),
            "dataset_from_index": pa.array(dataset_from, type=pa.int64()),
            "dataset_to_index": pa.array(dataset_to, type=pa.int64()),
            "meta/episodes/chunk_index": pa.array([0] * num_eps, type=pa.int64()),
            "meta/episodes/file_index": pa.array([0] * num_eps, type=pa.int64()),
        }

        # Add video-related columns
        for feat_key in self._image_keys:
            ep_parquet_columns[f"videos/{feat_key}/chunk_index"] = pa.array([0] * num_eps, type=pa.int64())
            ep_parquet_columns[f"videos/{feat_key}/file_index"] = pa.array(list(range(num_eps)), type=pa.int64())
            ep_parquet_columns[f"videos/{feat_key}/from_timestamp"] = pa.array(from_ts, type=pa.float32())
            ep_parquet_columns[f"videos/{feat_key}/to_timestamp"] = pa.array(to_ts, type=pa.float32())

        ep_pq_table = pa.table(ep_parquet_columns)
        pq.write_table(ep_pq_table, episodes_meta_dir / "file-000.parquet")

        # ------------------------------------------------------------------
        # recording_meta.json
        # ------------------------------------------------------------------
        camera_streams = []
        for feat_key in self._image_keys:
            is_depth = "depths" in feat_key
            cam_name = feat_key.rsplit(".", 1)[-1]  # e.g. "front" or "wrist"
            camera_streams.append({
                "name": cam_name,
                "feature_key": feat_key,
                "modality": "depth" if is_depth else "rgb",
                "resolution": {"width": img_w, "height": img_h},
                "frequency_hz": self._fps,
                "encoding_format": {
                    "storage_dtype": "video",
                    "video_codec": "ffv1" if is_depth else "h264",
                    "pixel_format": "gray16le" if is_depth else "yuv420p",
                    "encoder": {
                        "vcodec": "h264",
                        "pix_fmt": "yuv420p",
                        "g": None,
                        "crf": 23,
                        "preset": "fast",
                        "tune": None,
                    },
                },
            })

        recording_meta = {
            "schema_version": 1,
            "dataset": {
                "repo_id": f"pipi-grab-v1_{time.strftime('%Y%m%d_%H%M%S')}",
                "root": str(self._output_dir.resolve()),
                "fps": self._fps,
                "num_episodes": num_eps,
                "num_frames": self._total_frames,
                "total_tasks": 1,
            },
            "observation": {
                "camera_streams": camera_streams,
            },
        }

        with open(meta_dir / "recording_meta.json", "w") as fp:
            json.dump(recording_meta, fp, indent=2)

        # ------------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------------
        print(f"\nLeRobot dataset written to {self._output_dir.resolve()}")
        print(f"  Episodes: {num_eps}")
        print(f"  Total frames: {self._total_frames}")
        print(f"  FPS: {self._fps}")
        print(f"  State dim: {state_dim}")
        print(f"  Action dim: {action_dim}")
        if self._image_keys:
            print(f"  Camera features: {self._image_keys}")
