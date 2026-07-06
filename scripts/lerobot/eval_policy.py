#!/usr/bin/env python3
"""
Evaluate a trained ACT policy in the Piper Grab Visuomotor environment.

Usage::

    # Normal mode — open Isaac Sim with rendering, no video export:
    ./isaaclab.sh -p scripts/lerobot/eval_policy.py \\
        --checkpoint-dir ./datasets/lerobot/piper_grab_V1_D3_act_checkpoints/checkpoint_step_100000 \\
        --num-episodes 10 --device cuda:0

    # Headless + export test videos:
    ./isaaclab.sh -p scripts/lerobot/eval_policy.py \\
        --checkpoint-dir ./datasets/lerobot/piper_grab_V1_D3_act_checkpoints/checkpoint_step_100000 \\
        --num-episodes 10 --video ./datasets/eval_videos \\
        --headless --enable_cameras --device cuda:0
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.v2 as T

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate ACT policy on Piper Grab.")
parser.add_argument(
    "--checkpoint-dir",
    type=str,
    required=True,
    help="Path to ACT checkpoint directory (e.g., .../checkpoint_step_100000).",
)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Piper-Grab-IK-Rel-Visuomotor-v1",
    help="Isaac Lab task name.",
)
parser.add_argument("--num-episodes", type=int, default=10, help="Number of evaluation episodes.")
parser.add_argument("--max-steps", type=int, default=800, help="Max steps per episode.")
parser.add_argument(
    "--video",
    type=str,
    default=None,
    help="Output directory for per-episode MP4 videos. If not set, no video is recorded.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (must match training setup)
# ---------------------------------------------------------------------------
IMG_RESIZE = (224, 224)
CAM_KEY_MAP = {
    "table_cam": "observation.images.front",
    "wrist_cam": "observation.images.wrist",
}
STATE_KEY = "observation.state"

# Full state key order from training (must match the checkpoint's observation.state shape)
# Auto-detected: actual dims might differ from comments below
STATE_KEY_ORDER = [
    "actions",
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
# sum will be computed at runtime from the preprocessor stats

# ---------------------------------------------------------------------------
# Image transforms (matching training: resize + ImageNet norm)
# ---------------------------------------------------------------------------
def build_image_transforms():
    return T.Compose([
        T.Resize(IMG_RESIZE, antialias=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# ---------------------------------------------------------------------------
# Policy loading
# ---------------------------------------------------------------------------
def load_policy(checkpoint_dir: str, device: torch.device):
    ckpt_path = Path(checkpoint_dir)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    config_path = ckpt_path / "config.json"
    with open(config_path) as f:
        raw_config = json.load(f)

    input_features_raw = raw_config.pop("input_features", {})
    output_features_raw = raw_config.pop("output_features", {})
    raw_config.pop("type", None)

    cfg = ACTConfig(**raw_config)
    cfg.input_features = {
        k: PolicyFeature(type=FeatureType(v["type"]), shape=tuple(v["shape"]))
        for k, v in input_features_raw.items()
    }
    cfg.output_features = {
        k: PolicyFeature(type=FeatureType(v["type"]), shape=tuple(v["shape"]))
        for k, v in output_features_raw.items()
    }
    logger.info(f"Config loaded: input={list(cfg.input_features.keys())}, output={list(cfg.output_features.keys())}")

    import safetensors.torch as sft
    state_dict = sft.load_file(str(ckpt_path / "model.safetensors"), device=str(device))
    policy = ACTPolicy(cfg)
    policy.load_state_dict(state_dict)
    policy.to(device)
    policy.eval()
    logger.info(f"Policy loaded, device={device}")

    preprocessor, postprocessor = make_pre_post_processors(cfg, pretrained_path=str(ckpt_path))
    return policy, preprocessor, postprocessor

# ---------------------------------------------------------------------------
# Environment creation
# ---------------------------------------------------------------------------
def create_env(task_name: str, expected_action_dim: int):
    env_cfg = parse_env_cfg(task_name=task_name, device=args_cli.device, num_envs=1)

    # If checkpoint expects 8D action (old format: 6 IK + 2 gripper joint pos),
    # override gripper from 1D scalar to 2D joint position
    if expected_action_dim == 8:
        from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
        env_cfg.actions.gripper_action = JointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint7", "joint8"],
            scale=1.0,
        )
    # else: keep default 7D (6 IK + 1 gripper scalar via MimicBinaryJointPositionAction)

    env_cfg.gripper_joint_names = ["joint7", "joint8"]
    env_cfg.gripper_open_vals = [0.05, -0.05]
    env_cfg.gripper_threshold = 0.01
    env_cfg.recorders = None
    env_cfg.terminations = None

    env = gym.make(task_name, cfg=env_cfg).unwrapped
    logger.info(f"Env={task_name}, action_space={env.action_space} (expected={expected_action_dim}D)")
    return env

# ---------------------------------------------------------------------------
# Build state tensor from env observation
# ---------------------------------------------------------------------------
def build_state_tensor(obs_group: dict, expected_dim: int, device: torch.device) -> torch.Tensor:
    """Concatenate state keys into a single vector, zero-filling missing keys to match expected_dim."""
    parts = []
    for key in STATE_KEY_ORDER:
        val = obs_group.get(key)
        if val is not None and isinstance(val, torch.Tensor) and val.numel() > 0:
            parts.append(val.float().reshape(-1).to(device))
        else:
            parts.append(torch.empty(0, device=device))  # placeholder, will pad later
    # Concatenate what we have, then pad to expected_dim
    if parts:
        state = torch.cat(parts)
    else:
        state = torch.empty(0, device=device)
    if len(state) < expected_dim:
        padding = torch.zeros(expected_dim - len(state), device=device)
        state = torch.cat([state, padding])
    return state.unsqueeze(0)

# ---------------------------------------------------------------------------
# Build image batch
# ---------------------------------------------------------------------------
def build_image_batch(obs_group: dict, device: torch.device, transforms):
    batch = {}
    for obs_key, feat_key in CAM_KEY_MAP.items():
        val = obs_group.get(obs_key)
        if val is not None and isinstance(val, torch.Tensor):
            img = val.float()
            if img.ndim == 3:
                img = img.unsqueeze(0)  # (1, H, W, 3)
            if img.ndim == 4 and img.shape[-1] == 3:
                img = img.permute(0, 3, 1, 2)  # (1, 3, H, W)
            img = transforms(img.to(device))
            batch[feat_key] = img
        else:
            batch[feat_key] = torch.zeros(1, 3, *IMG_RESIZE, device=device)
    return batch

# ---------------------------------------------------------------------------
# Success check
# ---------------------------------------------------------------------------
def check_success(env, obs: dict) -> bool:
    subtask_terms = obs.get("subtask_terms", {})
    for key in ["placed_1", "placed_2"]:
        val = subtask_terms.get(key)
        if val is not None and isinstance(val, torch.Tensor) and val.numel() > 0:
            if bool(val[0].item()):
                return True
    try:
        mgr = getattr(env, "termination_manager", None)
        if mgr is not None:
            buf = mgr._term_buf.get("success")
            if buf is not None and buf.numel() > 0:
                return bool(buf[0].item())
    except Exception:
        pass
    return False

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = torch.device(args_cli.device)

    # 1. Load policy
    policy, preprocessor, postprocessor = load_policy(args_cli.checkpoint_dir, device)

    # Detect expected state/action dims from preprocessor safetensors
    import safetensors
    preproc_path = Path(args_cli.checkpoint_dir)
    expected_state_dim = 63
    expected_action_dim = 8
    for fname in sorted(preproc_path.glob("policy_preprocessor*.safetensors")):
        try:
            with safetensors.safe_open(str(fname), framework="pt") as sf:
                if "observation.state.mean" in sf.keys():
                    expected_state_dim = sf.get_tensor("observation.state.mean").shape[-1]
                if "action.mean" in sf.keys():
                    expected_action_dim = sf.get_tensor("action.mean").shape[-1]
        except Exception:
            pass
    logger.info(f"Expected dims from checkpoint: state={expected_state_dim}, action={expected_action_dim}")

    # Check: policy needs images → cameras must be enabled
    needs_images = bool(policy.config.image_features)
    if needs_images and not args_cli.enable_cameras:
        logger.warning("=" * 60)
        logger.warning("⚠️  Policy has visual features but --enable_cameras is NOT set!")
        logger.warning("   Cameras won't capture frames → policy gets zero images → blind inference.")
        logger.warning("   Add --enable_cameras to your command.")
        logger.warning("=" * 60)

    # 2. Create env (with action dim matching checkpoint)
    env = create_env(args_cli.task, expected_action_dim)
    image_transforms = build_image_transforms()

    # 3. Video setup
    video_dir = None
    if args_cli.video:
        video_dir = Path(args_cli.video)
        video_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Video output: {video_dir}")

    success_count = 0
    total_steps = 0

    for ep in range(args_cli.num_episodes):
        logger.info(f"{'='*50}")
        logger.info(f"Episode {ep + 1}/{args_cli.num_episodes}")

        obs, _ = env.reset()
        episode_steps = 0
        episode_frames = [] if video_dir else None

        while episode_steps < args_cli.max_steps:
            obs_group = obs.get("policy", obs)

            # --- Capture frame ---
            if episode_frames is not None:
                front = obs_group.get("table_cam")
                if front is not None and isinstance(front, torch.Tensor):
                    f = front.squeeze(0).cpu().numpy().astype(np.uint8)
                    episode_frames.append(f)

            # --- Build policy input ---
            batch = build_image_batch(obs_group, device, image_transforms)
            batch[STATE_KEY] = build_state_tensor(obs_group, expected_state_dim, device)

            # Preprocess
            batch = preprocessor(batch)
            if policy.config.image_features:
                batch = dict(batch)
                batch["observation.images"] = [batch[k] for k in policy.config.image_features]

            # Infer
            with torch.inference_mode():
                actions_hat, _ = policy.model(batch)

            action = actions_hat[:, 0, :].cpu()
            action = postprocessor(action).squeeze(0).numpy()

            # Debug: print first few actions to verify policy is producing non-zero output
            if episode_steps == 0:
                logger.info(f"  Action[0] = {np.array2string(action, precision=4, suppress_small=False)}")
            elif episode_steps == 1:
                logger.info(f"  Action[1] = {np.array2string(action, precision=4, suppress_small=False)}")

            # Step env
            env_action = torch.from_numpy(action).reshape(env.action_space.shape).to(env.device)
            obs, reward, terminated, truncated, info = env.step(env_action)

            episode_steps += 1
            total_steps += 1

            # Check termination
            done = bool(terminated.item()) if hasattr(terminated, "item") else bool(terminated)
            done = done or (bool(truncated.item()) if hasattr(truncated, "item") else bool(truncated))
            if done:
                break

        # --- Result ---
        ok = check_success(env, obs)
        if ok:
            success_count += 1
            logger.info(f"  ✅ Episode {ep + 1} SUCCESS ({episode_steps} steps)")
        elif episode_steps >= args_cli.max_steps:
            logger.info(f"  ⏱️  Episode {ep + 1} TIMEOUT ({episode_steps} steps)")
        else:
            logger.info(f"  ❌ Episode {ep + 1} FAILED ({episode_steps} steps)")

        # --- Save video ---
        if episode_frames:
            import torchvision.io as tio
            frames_tensor = torch.from_numpy(np.stack(episode_frames))
            vpath = video_dir / f"ep_{ep:03d}.mp4"
            tio.write_video(str(vpath), frames_tensor, fps=30)
            logger.info(f"  🎥 {vpath}")

    # 4. Summary
    logger.info(f"{'='*50}")
    logger.info(f"Done: {success_count}/{args_cli.num_episodes}  ({100 * success_count / max(1, args_cli.num_episodes):.1f}%)")
    logger.info(f"Total steps: {total_steps}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    try:
        while simulation_app.is_running():
            simulation_app.update()
    except KeyboardInterrupt:
        pass
    simulation_app.close()
