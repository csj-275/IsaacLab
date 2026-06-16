#!/usr/bin/env python3
# Copyright (c) 2024-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Evaluate a LeRobot-trained policy inside an Isaac Lab simulation environment.

This script bridges LeRobot's policy inference pipeline (preprocessor → model →
postprocessor) with Isaac Lab's observation/action interface.  It handles the
data-format translation between the two frameworks:

* Isaac Lab images are **NHWC uint8 [0,255]** ; LeRobot policies expect
  **NCHW float32 [0,1]** with batch dimension.
* Isaac Lab depth is **NHWC float32 meters** (1 channel) ; the LeRobot training
  pipeline stored depth as 3-channel MP4 video, so we replicate 1 → 3 channels
  to match the normalizer statistics.
* Isaac Lab state is a **dict of per-term tensors** ; LeRobot policies expect a
  single ``observation.state`` vector whose dimension must match the checkpoint.

Usage (inside the Docker container)::

    DISPLAY= CUDA_VISIBLE_DEVICES=0 ./isaaclab.sh -p \\
      scripts/imitation_learning/isaaclab_mimic/evaluate_lerobot_policy.py \\
      --task Isaac-Piper-Grab-IK-Rel-Visuomotor-Mimic-v1 \\
      --checkpoint /workspace/lerobot/outputs/act_piper_grab/checkpoints/002000/pretrained_model \\
      --num_episodes 10 --headless --enable_cameras --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Lerobot path — inside Docker, lerobot lives at /workspace/lerobot
# ---------------------------------------------------------------------------
_LEROBOT_PATHS = [
    "/workspace/lerobot/src",
    os.path.expanduser("~/company/lerobot/src"),
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "lerobot", "src"),
]
for _p in _LEROBOT_PATHS:
    _p = os.path.abspath(_p)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
        break

# ---------------------------------------------------------------------------
# Isaac Sim AppLauncher must be created BEFORE most Isaac Lab imports.
# ---------------------------------------------------------------------------
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Evaluate a LeRobot-trained policy in an Isaac Lab environment."
)
parser.add_argument("--task", type=str, required=True, help="Isaac Lab task name.")
parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="Path to LeRobot pretrained_model directory (contains config.json, model.safetensors, etc.).",
)
parser.add_argument("--num_episodes", type=int, default=10, help="Number of evaluation episodes.")
parser.add_argument("--max_steps", type=int, default=500, help="Max steps per episode.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--fps", type=int, default=30, help="Video recording FPS (if --record).")
parser.add_argument("--record", action="store_true", help="Record rollout videos.")
parser.add_argument("--output_dir", type=str, default="rollouts", help="Directory for recorded videos.")
parser.add_argument(
    "--no_depth",
    action="store_true",
    help="Skip depth observations (useful if checkpoint was trained without depth).",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# LeRobot imports — after AppLauncher to avoid pre-init conflicts
# ---------------------------------------------------------------------------
import gymnasium as gym

from isaaclab.envs import ManagerBasedRLEnv

import isaaclab_mimic.envs  # noqa: F401  — register Mimic envs
import isaaclab_tasks  # noqa: F401  — register task envs

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

logger = logging.getLogger(__name__)

# LeRobot policy/config imports
from lerobot.policies import make_policy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.batch_processor import AddBatchDimensionObservationStep
from lerobot.processor.device_processor import DeviceProcessorStep
from lerobot.processor.normalize_processor import NormalizerProcessorStep, UnnormalizerProcessorStep
from lerobot.processor.rename_processor import RenameObservationsProcessorStep


# ===========================================================================
# LeRobot-Compatible Observation Builder
# ===========================================================================


class LeRobotObservationBuilder:
    """Convert Isaac Lab policy-group observations into the dict expected by a
    LeRobot policy preprocessor.

    The builder is parameterised from the checkpoint so that it automatically
    adapts to different state dimensions and camera setups.
    """

    def __init__(
        self,
        input_features: dict[str, Any],
        device: torch.device,
    ) -> None:
        """
        Args:
            input_features: The ``input_features`` dict from the checkpoint
                ``config.json``.  Keys map to ``PolicyFeature``-like dicts with
                ``type`` and ``shape`` fields.
            device: Target device for tensors.
        """
        self._device = device
        self._features: dict[str, dict] = {}

        # Index features by LeRobot key
        for key, feat in input_features.items():
            self._features[key] = feat

        # Determine expected state dimension
        state_feat = self._features.get("observation.state")
        self._state_dim = state_feat["shape"][0] if state_feat else 0
        if self._state_dim > 0:
            logger.info(f"Expecting observation.state with {self._state_dim} dims")

        # Determine image feature keys and their declared shapes
        self._image_feature_keys: list[str] = []
        for k in sorted(self._features):
            if k.startswith("observation.images."):
                self._image_feature_keys.append(k)
            elif k.startswith("observation.depths."):
                self._image_feature_keys.append(k)

        if self._image_feature_keys:
            logger.info(f"Image/depth features: {self._image_feature_keys}")

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def image_feature_keys(self) -> list[str]:
        return self._image_feature_keys

    # ------------------------------------------------------------------
    # Camera name ↔ Isaac Lab obs key
    # ------------------------------------------------------------------
    @staticmethod
    def _isaac_cam_name(lerobot_feature_key: str) -> str:
        """LeRobot feature key → Isaac Lab observation key.

        ``observation.images.front`` → ``table_cam``
        ``observation.depths.wrist`` → ``wrist_cam_depth``
        """
        # e.g. "observation.images.front" → ["observation", "images", "front"]
        parts = lerobot_feature_key.split(".")
        modality = parts[1]  # "images" or "depths"
        cam = parts[2]  # "front" or "wrist"

        # Map LeRobot camera name → Isaac Lab name
        cam_map = {"front": "table_cam", "wrist": "wrist_cam"}
        isaac_cam = cam_map.get(cam, cam)

        if modality == "depths":
            return f"{isaac_cam}_depth"
        return isaac_cam

    # ------------------------------------------------------------------
    # State construction
    # ------------------------------------------------------------------
    def build_state(self, policy_obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Build ``observation.state`` from Isaac Lab policy observation dict.

        For a 7-dim state (6 arm joints + 1 gripper):
            ``joint_pos`` (first 6) + ``gripper_pos`` (first / mean) → (1, 7)
        """
        if self._state_dim == 0:
            return torch.zeros(1, 0, device=self._device)

        # Common case: 7-dim = 6 arm joints + 1 gripper
        if self._state_dim == 7 and "joint_pos" in policy_obs:
            jp = policy_obs["joint_pos"]  # (1, N) — may be 6, 7, or 8
            n_joints = jp.shape[-1]

            if n_joints >= 7:
                # Piper: 6 arm + 1 or 2 gripper → take first 6 + gripper mean
                arm = jp[:, :6]
                if "gripper_pos" in policy_obs:
                    gripper = policy_obs["gripper_pos"].float().mean(dim=-1, keepdim=True)
                else:
                    gripper = jp[:, 6:7]  # single gripper dim
                return torch.cat([arm.float(), gripper], dim=-1)  # (1, 7)
            else:
                # Pad with zeros if needed
                padded = torch.zeros(1, self._state_dim, device=jp.device, dtype=torch.float32)
                padded[:, :n_joints] = jp.float()
                return padded.to(self._device)

        # Generic fallback: concatenate all non-image state keys
        state_parts: list[torch.Tensor] = []
        # NOTE: "actions" is excluded to avoid a feedback loop
        STATE_LIKE_KEYS = {
            "joint_pos", "joint_vel", "eef_pos", "eef_quat",
            "gripper_pos", "object",
            "object_1_positions", "object_1_orientations",
            "box_positions", "box_orientations",
            "mug_positions", "mug_orientations",
        }
        for k in sorted(policy_obs):
            if k in STATE_LIKE_KEYS and not self._is_image(policy_obs[k]):
                state_parts.append(policy_obs[k].float())

        if not state_parts:
            logger.warning("No state keys found in policy obs; returning zero state")
            return torch.zeros(1, self._state_dim, device=self._device)

        state = torch.cat(state_parts, dim=-1)
        if state.shape[-1] != self._state_dim:
            logger.warning(
                f"Built state has {state.shape[-1]} dims, expected {self._state_dim}. "
                f"Truncating/padding."
            )
            if state.shape[-1] > self._state_dim:
                state = state[:, : self._state_dim]
            else:
                pad = torch.zeros(1, self._state_dim - state.shape[-1], device=state.device)
                state = torch.cat([state, pad], dim=-1)
        return state.to(self._device)

    # ------------------------------------------------------------------
    # Image conversion
    # ------------------------------------------------------------------
    @staticmethod
    def _is_image(tensor: torch.Tensor) -> bool:
        return tensor.ndim >= 3

    @staticmethod
    def convert_rgb(tensor: torch.Tensor) -> torch.Tensor:
        """NHWC uint8 [0,255] → NCHW float32 [0,1]."""
        if tensor.dtype == torch.uint8:
            tensor = tensor.float() / 255.0
        elif tensor.max() > 1.1:  # likely [0,255] range as float
            tensor = tensor.float() / 255.0
        tensor = tensor.to(dtype=torch.float32)
        # NHWC → NCHW
        if tensor.ndim == 4 and tensor.shape[-1] in (1, 3, 4):
            tensor = tensor.permute(0, 3, 1, 2)
        elif tensor.ndim == 3 and tensor.shape[-1] in (1, 3, 4):
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        return tensor

    @staticmethod
    def convert_depth(tensor: torch.Tensor) -> torch.Tensor:
        """NHWC float32 meters (1-ch) → NCHW float32 [0,1] (3-ch).

        During training, depth was stored as 3-channel MP4 video via
        ``_normalize_depth_for_video`` which does per-episode min/max
        normalization to [0, 255] uint8, then replicates to 3 channels.
        The LeRobot dataloader then reads uint8 [0, 255], divides by 255
        to get [0, 1] float32 in NCHW.

        At inference time we approximate this with a fixed depth range
        of [0, 5] meters (clip → scale → replicate → NCHW).  The MEAN_STD
        normalizer step will handle the rest.
        """
        MAX_DEPTH_M = 5.0  # clip depth beyond 5 metres

        tensor = tensor.float()

        # Handle (1, H, W) case (no channel dim)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(-1)  # → (1, H, W, 1)

        # Replace inf/nan with max depth
        tensor = torch.nan_to_num(tensor, nan=MAX_DEPTH_M, posinf=MAX_DEPTH_M)

        # Clip to [0, MAX_DEPTH_M] then scale to [0, 1]
        tensor = torch.clamp(tensor, 0.0, MAX_DEPTH_M) / MAX_DEPTH_M

        # Replicate 1 → 3 channels (matches MP4 RGB encoding from training;
        # normalizer stats for depth are shape [3, 1, 1] confirmed from
        # safetensors header)
        if tensor.shape[-1] == 1:
            tensor = tensor.repeat(1, 1, 1, 3)

        # NHWC → NCHW
        tensor = tensor.permute(0, 3, 1, 2)
        return tensor.to(dtype=torch.float32)

    # ------------------------------------------------------------------
    # Build full LeRobot observation dict
    # ------------------------------------------------------------------
    def build(self, policy_obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Convert Isaac Lab ``policy`` group observations → LeRobot dict.

        Returns a flat dict with keys like ``observation.state``,
        ``observation.images.front``, ``observation.depths.wrist``, etc.
        All tensors are on the target device, float32, NCHW for images.
        """
        result: dict[str, torch.Tensor] = {}

        # State
        if self._state_dim > 0:
            result["observation.state"] = self.build_state(policy_obs)

        # Images
        for feat_key in self._image_feature_keys:
            isaac_key = self._isaac_cam_name(feat_key)
            if isaac_key not in policy_obs:
                logger.warning(f"Camera obs key '{isaac_key}' not found in policy obs; skipping {feat_key}")
                continue

            tensor = policy_obs[isaac_key]
            is_depth = "depths" in feat_key or "depth" in isaac_key

            if is_depth:
                result[feat_key] = self.convert_depth(tensor)
            else:
                result[feat_key] = self.convert_rgb(tensor)

        return result


# ===========================================================================
# Policy loading
# ===========================================================================


def load_policy_and_processors(
    checkpoint_dir: str | Path,
    device: torch.device,
) -> tuple[Any, PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """Load a LeRobot policy checkpoint with its pre/post processors.

    Returns:
        (policy, preprocessor, postprocessor)
    """
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    if not (checkpoint_dir / "config.json").exists():
        raise FileNotFoundError(f"config.json not found in {checkpoint_dir}")

    logger.info(f"Loading policy from {checkpoint_dir}")

    # Load policy using LeRobot's factory
    policy = make_policy(str(checkpoint_dir))
    policy.eval()
    policy.to(device)

    # Build preprocessor from saved pipeline config
    preprocessor_cfg_path = checkpoint_dir / "policy_preprocessor.json"
    postprocessor_cfg_path = checkpoint_dir / "policy_postprocessor.json"

    preprocessor = _build_processor(preprocessor_cfg_path, checkpoint_dir, device)
    postprocessor = _build_processor(postprocessor_cfg_path, checkpoint_dir, device)

    logger.info("Policy and processors loaded successfully")
    return policy, preprocessor, postprocessor


def _build_processor(
    config_path: Path,
    checkpoint_dir: Path,
    device: torch.device,
) -> PolicyProcessorPipeline:
    """Build a PolicyProcessorPipeline from a saved config file.

    We manually construct the pipeline steps rather than using the full
    LeRobot deserialization machinery because we need to control device
    placement and avoid dependency on the training config parsing.
    """
    with open(config_path) as f:
        proc_cfg = json.load(f)

    steps: list[ProcessorStep] = []

    for step_cfg in proc_cfg.get("steps", []):
        registry_name = step_cfg["registry_name"]
        config = step_cfg.get("config", {})

        if registry_name == "rename_observations_processor":
            steps.append(RenameObservationsProcessorStep(config.get("rename_map", {})))

        elif registry_name == "to_batch_processor":
            steps.append(AddBatchDimensionObservationStep())

        elif registry_name == "device_processor":
            steps.append(
                DeviceProcessorStep(
                    device=device,
                    float_dtype=config.get("float_dtype"),
                )
            )

        elif registry_name == "normalizer_processor":
            # Load stats from the safetensors file
            state_file = step_cfg.get("state_file")
            if state_file:
                state_path = checkpoint_dir / state_file

                steps.append(
                    NormalizerProcessorStep(
                        features=config["features"],
                        norm_map=config["norm_map"],
                        stats_path=str(state_path),
                        eps=config.get("eps", 1e-8),
                    )
                )
            else:
                logger.warning("Normalizer step has no state_file; skipping")

        elif registry_name == "unnormalizer_processor":
            state_file = step_cfg.get("state_file")
            if state_file:
                state_path = checkpoint_dir / state_file
                steps.append(
                    UnnormalizerProcessorStep(
                        features=config["features"],
                        norm_map=config["norm_map"],
                        stats_path=str(state_path),
                        eps=config.get("eps", 1e-8),
                    )
                )

        else:
            logger.warning(f"Unknown processor step: {registry_name}; skipping")

    return PolicyProcessorPipeline(steps)


# ===========================================================================
# Main evaluation loop
# ===========================================================================


def main():
    # ------------------------------------------------------------------
    # Load checkpoint metadata
    # ------------------------------------------------------------------
    checkpoint_dir = Path(args_cli.checkpoint)
    with open(checkpoint_dir / "config.json") as f:
        policy_config = json.load(f)

    input_features = policy_config.get("input_features", {})
    output_features = policy_config.get("output_features", {})
    n_action_steps = policy_config.get("n_action_steps", 50)
    device = torch.device(args_cli.device if args_cli.device else "cuda")

    # ------------------------------------------------------------------
    # Load policy
    # ------------------------------------------------------------------
    policy, preprocessor, postprocessor = load_policy_and_processors(
        checkpoint_dir, device
    )

    # ------------------------------------------------------------------
    # Create observation builder
    # ------------------------------------------------------------------
    obs_builder = LeRobotObservationBuilder(input_features, device)

    # ------------------------------------------------------------------
    # Create environment
    # ------------------------------------------------------------------
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=1
    )

    # Disable concatenation → per-term dict observations
    env_cfg.observations.policy.concatenate_terms = False

    # Disable terminations — we handle success tracking manually
    env_cfg.terminations = None

    # Disable recorders
    env_cfg.recorders = None

    # Set seed
    env_cfg.seed = args_cli.seed

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    if not isinstance(env, ManagerBasedRLEnv):
        raise TypeError(
            f"Expected ManagerBasedRLEnv, got {type(env)}. "
            "Use a task registered as ManagerBasedRLEnv."
        )

    # ------------------------------------------------------------------
    # Optional video recording setup
    # ------------------------------------------------------------------
    record = args_cli.record
    if record:
        output_dir = Path(args_cli.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        video_writers: dict[str, Any] = {}
        logger.info(f"Recording videos to {output_dir.resolve()}")

    # ------------------------------------------------------------------
    # Evaluation loop
    # ------------------------------------------------------------------
    success_count = 0
    total_steps = 0

    logger.info(
        f"Starting evaluation: {args_cli.num_episodes} episodes, "
        f"max {args_cli.max_steps} steps each"
    )

    for ep in range(args_cli.num_episodes):
        obs_dict, _ = env.reset()
        policy.reset()

        episode_steps = 0
        episode_success = False

        with torch.inference_mode():
            for step in range(args_cli.max_steps):
                # ---- Build LeRobot observation -----------------------------------
                lerobot_obs = obs_builder.build(obs_dict["policy"])

                # ---- Preprocess → infer → postprocess ---------------------------
                lerobot_obs = preprocessor(lerobot_obs)
                action = policy.select_action(lerobot_obs)
                action = postprocessor({"action": action})
                env_action = action["action"].to(device)

                # ---- Step environment -------------------------------------------
                obs_dict, _, terminated, truncated, _ = env.step(env_action)
                episode_steps += 1

                # ---- Check success via subtask terms ----------------------------
                subtask_terms = obs_dict.get("subtask_terms", {})
                if subtask_terms.get("placed_1", torch.tensor([False]))[0]:
                    episode_success = True

                # ---- Optional recording -----------------------------------------
                if record:
                    _record_frame(obs_dict, video_writers, output_dir, ep, step, args_cli.fps)

                if terminated or truncated:
                    break

        total_steps += episode_steps
        if episode_success:
            success_count += 1

        logger.info(
            f"Episode {ep + 1:3d}/{args_cli.num_episodes}  "
            f"steps={episode_steps:3d}  success={'✓' if episode_success else '✗'}"
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    success_rate = 100.0 * success_count / args_cli.num_episodes
    avg_steps = total_steps / args_cli.num_episodes
    print(f"\n{'=' * 50}")
    print(f"Evaluation complete")
    print(f"  Episodes:     {args_cli.num_episodes}")
    print(f"  Successes:    {success_count} ({success_rate:.1f}%)")
    print(f"  Avg steps:    {avg_steps:.0f}")
    print(f"{'=' * 50}")

    # Cleanup
    if record:
        for w in video_writers.values():
            w.release()

    env.close()


def _record_frame(
    obs_dict: dict,
    writers: dict,
    output_dir: Path,
    episode: int,
    step: int,
    fps: int,
) -> None:
    """Record a single frame to MP4 video writers (lazy-init)."""
    import cv2

    policy_obs: dict = obs_dict.get("policy", {})
    for key, tensor in policy_obs.items():
        if tensor.ndim < 4:
            continue  # skip non-image keys

        frame = tensor[0]  # first (only) env
        frame_np = frame.cpu().numpy()

        # Ensure uint8 3-channel
        if frame_np.dtype != np.uint8:
            frame_np = (frame_np * 255).astype(np.uint8) if frame_np.max() <= 1.0 else frame_np.astype(np.uint8)
        if frame_np.ndim == 3 and frame_np.shape[-1] == 1:
            frame_np = np.repeat(frame_np, 3, axis=-1)

        # Lazy-init writer
        if key not in writers:
            video_dir = output_dir / f"episode_{episode:04d}"
            video_dir.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            h, w = frame_np.shape[:2]
            writers[key] = cv2.VideoWriter(str(video_dir / f"{key}.mp4"), fourcc, fps, (w, h))

        writers[key].write(cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        simulation_app.close()
