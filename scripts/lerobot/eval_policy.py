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

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless/sim
import matplotlib.pyplot as plt
import numpy as np
import torch

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
    default="Isaac-Piper-Grab-IK-Rel-Visuomotor-v1-A",
    help="Isaac Lab task name. V1-A uses joint position actions matching the trained policy.",
)
parser.add_argument("--num-episodes", type=int, default=10, help="Number of evaluation episodes.")
parser.add_argument("--max-steps", type=int, default=800, help="Max steps per episode.")
parser.add_argument(
    "--post-success-delay",
    type=int,
    default=0,
    help="Extra frames to continue after success before ending episode. 0 = end immediately.",
)
parser.add_argument(
    "--video",
    type=str,
    default=None,
    help="Output directory for per-episode MP4 videos. If not set, no video is recorded.",
)
parser.add_argument(
    "--plot-dir",
    type=str,
    default="logs/eval_plots",
    help="Output directory for state/action plots. Default: logs/eval_plots.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

# Random seed for reproducibility (edit if needed)
EVAL_SEED = 42

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
CAM_KEY_MAP = {
    "table_cam": "observation.images.front",
    "wrist_cam": "observation.images.wrist",
}
STATE_KEY = "observation.state"

# State key(s) to extract from observation. `build_state_tensor` auto-detects
# available keys and pads/truncates to match the checkpoint's expected dim.
# V1-A env uses "state" (joint_pos_target_7d). Legacy envs have "actions" etc.
STATE_KEY_ORDER = [
    "joint_pos_7d",         # V1 base: current joint positions
    # "joint_pos_target_7d",  # target joint pos (closes loop: model output → target → next state)
    # "state",                # fallback
    # "joint_pos",            # alternative key
    # "actions",              # legacy: last_action
]

# ---------------------------------------------------------------------------
# Image transforms — training had image_transforms.enable=False (default),
# so NO resize or augmentation was applied. Images go to the model as float32
# in [0, 1] range (training used return_uint8=false). The ResNet backbone uses
# AdaptiveAvgPool2d so it handles original sizes.
# ---------------------------------------------------------------------------
def build_image_transforms():
    return None  # no transforms — matches training setup

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

    # Remove training metadata keys not accepted by ACTConfig
    _TRAINING_KEYS = {
        "type", "pretrained_path", "pretrained_revision", "push_to_hub",
        "repo_id", "private", "tags", "license", "device", "use_amp", "use_peft",
    }
    for k in _TRAINING_KEYS:
        raw_config.pop(k, None)

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

    # Extract expected dims directly from config (more reliable than reading safetensors)
    state_ft = cfg.input_features.get(STATE_KEY)
    expected_state_dim = state_ft.shape[0] if state_ft is not None else 63
    action_ft = cfg.output_features.get("action")
    expected_action_dim = action_ft.shape[0] if action_ft is not None else 8

    logger.info(f"Policy loaded, device={device}")
    return policy, preprocessor, postprocessor, expected_state_dim, expected_action_dim

# ---------------------------------------------------------------------------
# Environment creation
# ---------------------------------------------------------------------------
def create_env(task_name: str, expected_action_dim: int):
    """Create evaluation environment.

    Uses V1-A task config which directly accepts joint position actions
    (matching the ACT policy's output format), as opposed to the regular
    V1 config which uses IK delta pose actions for data collection.

    The V1-A gripper uses BinaryJointPositionActionCfg (sign-based threshold=0),
    but the policy outputs continuous gripper positions (0.02-0.05).
    We override with AbsBinaryJointPositionActionCfg which uses a configurable
    threshold (0.035 = midpoint between open 0.05 and close 0.02 in training data).
    """
    env_cfg = parse_env_cfg(task_name=task_name, device=args_cli.device, num_envs=1)

    # Override gripper: use absolute binary action with threshold matching training data
    # Open=0.05, Close=0.02, midpoint=0.035
    from isaaclab.envs.mdp.actions.actions_cfg import AbsBinaryJointPositionActionCfg
    env_cfg.actions.gripper_action = AbsBinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint7", "joint8"],
        open_command_expr={"joint7": 0.05, "joint8": -0.05},
        close_command_expr={"joint7": -0.05, "joint8": 0.05},
        threshold=0.03,
        positive_threshold=True,  # grip > 0.03 → open; grip ≤ 0.03 → close
    )

    # Handle 8D action (old format: 6 joint pos + 2 gripper joint pos)
    if expected_action_dim == 8:
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
        )

    env_cfg.recorders = None

    env = gym.make(task_name, cfg=env_cfg).unwrapped
    logger.info(f"Env={task_name}, action_space={env.action_space} (expected={expected_action_dim}D)")
    return env

# ---------------------------------------------------------------------------
# Build state tensor from env observation
# ---------------------------------------------------------------------------
# Known image observation keys (not state)
# _IMAGE_OBS_KEYS = {"table_cam", "wrist_cam", "table_cam_depth", "wrist_cam_depth"}
# # Subtask / non-state keys
# _NON_STATE_KEYS = _IMAGE_OBS_KEYS | {"subtask_terms", "policy"}


def build_state_tensor(obs_group: dict, expected_dim: int, device: torch.device) -> torch.Tensor:
    """Extract the state tensor from observation by matching STATE_KEY_ORDER."""
    # Find the first available key in STATE_KEY_ORDER
    chosen_key = None
    for candidate in STATE_KEY_ORDER:
        val = obs_group.get(candidate)
        if val is not None and isinstance(val, torch.Tensor) and val.numel() > 0:
            chosen_key = candidate
            break

    if chosen_key is not None:
        state = obs_group[chosen_key].float().reshape(-1).to(device)
    else:
        available = [k for k, v in obs_group.items() if isinstance(v, torch.Tensor)]
        print(f"[WARN] build_state_tensor: NO KEY FOUND! Available tensor keys: {available}", flush=True)
        state = torch.empty(0, device=device)

    actual_dim = len(state)
    if actual_dim < expected_dim:
        padding = torch.zeros(expected_dim - actual_dim, device=device)
        state = torch.cat([state, padding])
    elif actual_dim > expected_dim:
        state = state[:expected_dim]

    return state.unsqueeze(0)

# ---------------------------------------------------------------------------
# Build image batch
# ---------------------------------------------------------------------------
def _simulate_mp4_roundtrip(img_uint8: np.ndarray) -> np.ndarray:
    """Encode a single uint8 RGB frame as MP4 then decode back, simulating training data pipeline.

    Tries ffmpeg first, falls back to OpenCV ``mp4v`` codec (same as LeRobot handler).
    """
    import subprocess, tempfile, os, cv2
    h, w = img_uint8.shape[:2]
    fd, tmppath = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        # Try ffmpeg first
        try:
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "rawvideo", "-vcodec", "rawvideo",
                "-s", f"{w}x{h}", "-pix_fmt", "rgb24", "-r", "30",
                "-i", "-",
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-pix_fmt", "yuv420p", "-frames:v", "1", tmppath,
            ]
            subprocess.run(cmd, input=img_uint8.tobytes(), capture_output=True, timeout=10, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            # Fallback: OpenCV mp4v (same as LeRobot handler's _encode_video_cv2)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(tmppath, fourcc, 30.0, (w, h))
            writer.write(cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))
            writer.release()
        # Decode back
        cap = cv2.VideoCapture(tmppath)
        ret, frame_bgr = cap.read()
        cap.release()
        if ret:
            return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return img_uint8
    except Exception:
        return img_uint8
    finally:
        try:
            os.unlink(tmppath)
        except OSError:
            pass


def build_image_batch(obs_group: dict, device: torch.device, transforms):
    """Convert env images (uint8 0-255) to float32 0-1, then optionally resize.

    Training used --dataset.use_imagenet_stats=false and return_uint8=false,
    so images were float32 0-1 with NO ImageNet normalization.
    We must divide by 255 to match the training data range.
    """
    batch = {}
    for obs_key, feat_key in CAM_KEY_MAP.items():
        val = obs_group.get(obs_key)
        if val is not None and isinstance(val, torch.Tensor):
            img_u8 = val.squeeze(0).cpu().numpy().astype(np.uint8)
            img_u8 = _simulate_mp4_roundtrip(img_u8)  # match training MP4 compression
            img = torch.from_numpy(img_u8).float() / 255.0
            img = img.unsqueeze(0)  # (1, H, W, 3)
            if img.ndim == 4 and img.shape[-1] == 3:
                img = img.permute(0, 3, 1, 2)  # (1, 3, H, W)
            img = img.to(device)
            if transforms is not None:
                img = transforms(img)
            batch[feat_key] = img
        else:
            batch[feat_key] = torch.zeros(1, 3, 720, 1280, device=device)
    return batch

# ---------------------------------------------------------------------------
# Success check
# ---------------------------------------------------------------------------
def _dummy_success_term(env, **kwargs) -> torch.Tensor:
    """Dummy success term that always returns False — used to disable auto-reset."""
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


# ---------------------------------------------------------------------------
# Video writing (OpenCV — avoids torchvision deprecation + ffmpeg pipe deadlock)
# ---------------------------------------------------------------------------
def check_success(env, obs: dict) -> bool:
    """Check if episode succeeded.

    Uses env's termination_manager.get_term("success") — the authoritative
    multi-stage success condition (e.g., objects_a_and_b_are_into_c).
    No subtask fallback to avoid false positives from partial completion.
    """
    try:
        mgr = getattr(env, "termination_manager", None)
        if mgr is not None and "success" in mgr.active_terms:
            return bool(mgr.get_term("success")[0].item())
    except Exception:
        pass
    return False


def _write_video_cv2(frames: list, video_dir, episode_index: int, fps: int = 30):
    """Write a list of uint8 numpy frames (H, W, 3) to MP4 via OpenCV."""
    import cv2
    vpath = Path(video_dir) / f"ep_{episode_index:03d}.mp4"
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(vpath), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("cv2.VideoWriter failed to open")
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    logger.info(f"  🎥 {vpath} ({len(frames)} frames)")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = torch.device(args_cli.device)

    # 1. Load policy
    policy, preprocessor, postprocessor, expected_state_dim, expected_action_dim = load_policy(
        args_cli.checkpoint_dir, device
    )
    logger.info(f"Expected dims from config: state={expected_state_dim}, action={expected_action_dim}")

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
    env.seed(EVAL_SEED)
    image_transforms = build_image_transforms()

    # 3. Video & plot setup
    video_dir = None
    if args_cli.video:
        video_dir = Path(args_cli.video)
        video_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Video output: {video_dir}")

    plot_dir = Path(args_cli.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Plot output: {plot_dir}")

    success_count = 0
    total_steps = 0
    joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]

    for ep in range(args_cli.num_episodes):
        logger.info(f"{'='*50}")
        logger.info(f"Episode {ep + 1}/{args_cli.num_episodes}")
        print(f"\n=== Episode {ep + 1}/{args_cli.num_episodes} ===", flush=True)

        obs, _ = env.reset()
        episode_steps = 0
        episode_frames = [] if video_dir else None
        ep_states: list[np.ndarray] = []
        ep_actions: list[np.ndarray] = []

        # One-time debug: verify observation structure
        if ep == 0:
            print(f"[DEBUG] obs top-level keys: {list(obs.keys())}", flush=True)
            policy_obs = obs.get("policy", obs)
            if isinstance(policy_obs, dict):
                print(f"[DEBUG] policy obs keys: {list(policy_obs.keys())}", flush=True)
                for k, v in policy_obs.items():
                    if isinstance(v, torch.Tensor):
                        if v.numel() <= 10:
                            print(f"[DEBUG]   {k}: shape={list(v.shape)}, values={v.squeeze().tolist()}", flush=True)
                        else:
                            print(f"[DEBUG]   {k}: shape={list(v.shape)}, min={v.min().item():.4f} max={v.max().item():.4f}", flush=True)
            print(f"[DEBUG] env.action_space.shape={env.action_space.shape}", flush=True)

        # ACT chunk execution: how many frames from the predicted chunk to execute
        # before re-inferring. None = execute the full chunk (100 frames).
        # Set to a smaller number (e.g., 10) for more frequent re-inference.
        CHUNK_EXEC_STEPS = 50  # None = full chunk; or int like 10, 25, 50

        # ACT chunk state — persists across steps within one episode
        action_chunk = None
        chunk_pos = 0
        chunk_len = 0

        # Post-success countdown (--post-success-delay): 0 = inactive
        post_success_delay = 0
        episode_success = False  # remember success even if term is later disabled

        while episode_steps < args_cli.max_steps or post_success_delay > 0:
            obs_group = obs.get("policy", obs)

            # --- Capture frame ---
            if episode_frames is not None:
                front = obs_group.get("table_cam")
                if front is not None and isinstance(front, torch.Tensor):
                    f = front.squeeze(0).cpu().numpy().astype(np.uint8)
                    episode_frames.append(f)

            # --- Build policy input ---
            batch = build_image_batch(obs_group, device, image_transforms)

            # One-time check: are images actually captured?
            if ep == 0 and episode_steps == 0:
                for k in ["observation.images.front", "observation.images.wrist"]:
                    img = batch.get(k)
                    if img is not None and isinstance(img, torch.Tensor):
                        print(f"[DEBUG] {k}: shape={list(img.shape)}, min={img.min().item():.4f}, max={img.max().item():.4f}, mean={img.mean().item():.4f}", flush=True)
                        if img.max().item() == 0.0:
                            print(f"[WARN] ⚠️  {k} is ALL ZEROS — cameras may not be rendering! Use --enable_cameras.", flush=True)
            raw_state = build_state_tensor(obs_group, expected_state_dim, device)
            batch[STATE_KEY] = raw_state

            # Preprocess
            batch = preprocessor(batch)
            if policy.config.image_features:
                batch = dict(batch)
                batch["observation.images"] = [batch[k] for k in policy.config.image_features]

            # One-time debug: show normalized values
            if ep == 0 and episode_steps == 0:
                ns = batch.get(STATE_KEY)
                if ns is not None:
                    print(f"[DEBUG] normalized state: shape={list(ns.shape)}, values={ns.squeeze().tolist()[:10]}", flush=True)
                for ik, imgk in enumerate(policy.config.image_features):
                    img = batch.get(imgk)
                    if img is not None:
                        print(f"[DEBUG] normalized {imgk}: shape={list(img.shape)}, min={img.min().item():.4f}, max={img.max().item():.4f}, mean={img.mean().item():.4f}", flush=True)

            # --- Re-infer when chunk exhausted or exec horizon reached ---
            need_reinfer = action_chunk is None or chunk_pos >= chunk_len
            if CHUNK_EXEC_STEPS is not None and chunk_pos >= CHUNK_EXEC_STEPS:
                need_reinfer = True
            if need_reinfer:
                with torch.inference_mode():
                    actions_hat, _ = policy.model(batch)
                # Full chunk: (1, n_action_steps, 7)
                action_chunk = actions_hat[:, :, :].cpu()  # (1, T, 7)
                chunk_len = action_chunk.shape[1]
                chunk_pos = 0

            # Take next action from chunk
            action = action_chunk[:, chunk_pos, :].clone()
            action = postprocessor(action).squeeze(0).numpy()
            chunk_pos += 1

            # Record state & action
            state_np = raw_state.squeeze(0).cpu().numpy()
            ep_states.append(state_np.copy())
            ep_actions.append(action.copy())

            # Step env
            env_action = torch.from_numpy(action).reshape(env.action_space.shape).to(env.device)
            obs, reward, terminated, truncated, info = env.step(env_action)

            # One-time debug: verify action → joint target mapping
            if ep == 0 and episode_steps == 0:
                robot = env.scene["robot"]
                jt = robot.data.joint_pos_target[0].cpu().numpy()
                jp = robot.data.joint_pos[0].cpu().numpy()
                print(f"[DEBUG] after step 0: joint_pos={np.array2string(jp, precision=3, suppress_small=True)}", flush=True)
                print(f"[DEBUG] after step 0: joint_target={np.array2string(jt, precision=3, suppress_small=True)}", flush=True)
                print(f"[DEBUG] after step 0: model_action={np.array2string(action, precision=3, suppress_small=True)}", flush=True)
                print(f"[DEBUG] chunk_len={chunk_len}, executing {min(10, chunk_len)} steps per inference", flush=True)

            episode_steps += 1
            total_steps += 1

            # Per-step progress
            # s_str = np.array2string(state_np, precision=2, suppress_small=True, max_line_width=60)
            # a_str = np.array2string(action, precision=2, suppress_small=True, max_line_width=80)
            # print(f"[Ep {ep + 1}/{args_cli.num_episodes}] Step {episode_steps}/{args_cli.max_steps} | s={s_str} | a={a_str}", flush=True)

            # Check termination
            if args_cli.post_success_delay > 0:
                if post_success_delay == 0 and check_success(env, obs):
                    # Disable success term to prevent env auto-reset during delay
                    episode_success = True
                    post_success_delay = args_cli.post_success_delay
                    mgr = env.termination_manager
                    term_idx = mgr._term_name_to_term_idx["success"]
                    mgr._term_cfgs[term_idx].func = _dummy_success_term
                    print(
                        f"[INFO] Success! Disabled success term, continuing {post_success_delay} frames...",
                        flush=True,
                    )

                if post_success_delay > 0:
                    post_success_delay -= 1
                    if post_success_delay == 0:
                        break
                else:
                    done = bool(terminated.item()) if hasattr(terminated, "item") else bool(terminated)
                    done = done or (bool(truncated.item()) if hasattr(truncated, "item") else bool(truncated))
                    if done:
                        break
            else:
                done = bool(terminated.item()) if hasattr(terminated, "item") else bool(terminated)
                done = done or (bool(truncated.item()) if hasattr(truncated, "item") else bool(truncated))
                if done:
                    episode_success = check_success(env, obs)
                    break

        # --- Result ---
        if episode_success:
            success_count += 1
            print(f"  ✅ Episode {ep + 1} SUCCESS ({episode_steps} steps)", flush=True)
        elif episode_steps >= args_cli.max_steps:
            print(f"  ⏱️  Episode {ep + 1} TIMEOUT ({episode_steps} steps)", flush=True)
        else:
            print(f"  ❌ Episode {ep + 1} FAILED ({episode_steps} steps)", flush=True)

        # --- Plot state/action ---
        if ep_states and ep_actions:
            states_arr = np.array(ep_states)   # (T, 7)
            actions_arr = np.array(ep_actions)  # (T, 7)
            steps = np.arange(len(ep_states))

            fig, axes = plt.subplots(4, 2, figsize=(16, 14))
            axes = axes.flatten()
            for j in range(7):
                ax = axes[j]
                ax.plot(steps, actions_arr[:, j], "-",  linewidth=1.5, label="action", color="C0")
                ax.plot(steps, states_arr[:, j],  "--", linewidth=1.5, label="state",  color="C1")
                ax.set_title(joint_names[j])
                ax.set_xlabel("step")
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=7)
            # Hide extra subplot
            axes[7].set_visible(False)

            status = "SUCCESS" if episode_success else ("TIMEOUT" if episode_steps >= args_cli.max_steps else "FAILED")
            fig.suptitle(f"Episode {ep + 1}  —  {status}  ({episode_steps} steps)", fontsize=14)
            fig.tight_layout()

            plot_path = plot_dir / f"ep_{ep:03d}.png"
            fig.savefig(str(plot_path), dpi=100)
            plt.close(fig)
            logger.info(f"  📊 {plot_path}")

        # --- Save video ---
        num_frames = len(episode_frames) if episode_frames else 0
        print(f"[DEBUG] episode_frames: {num_frames}", flush=True)
        if num_frames > 0:
            print("[DEBUG] Writing video...", flush=True)
            _write_video_cv2(episode_frames, video_dir, ep, fps=30)
            print("[DEBUG] Video done.", flush=True)
        elif video_dir is not None:
            logger.warning(f"  ⚠️  Episode {ep + 1}: 0 frames captured — check if 'table_cam' is in obs keys")

    # 4. Summary
    logger.info(f"{'='*50}")
    logger.info(f"Done: {success_count}/{args_cli.num_episodes}  ({100 * success_count / max(1, args_cli.num_episodes):.1f}%)")
    logger.info(f"Total steps: {total_steps}")
    logger.info(f"Plots saved to: {plot_dir}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as e:
        import traceback
        print(f"[ERROR] {e}", flush=True)
        traceback.print_exc()
    finally:
        import os
        print("[INFO] Done eval.", flush=True)
        os._exit(0)
