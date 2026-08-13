#!/usr/bin/env python3
"""
Evaluate a trained PI0.5 (pi05) policy in the Piper Grab Visuomotor environment.

Usage::

    # Headless + export test videos:
    ./isaaclab.sh -p scripts/lerobot/eval_pi_policy.py \\
        --checkpoint-dir logs/PI/checkpoints/010000/pretrained_model \\
        --num-episodes 10 --video logs/eval_videos \\
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

parser = argparse.ArgumentParser(description="Evaluate PI0.5 policy on Piper Grab.")
parser.add_argument(
    "--checkpoint-dir",
    type=str,
    required=True,
    help="Path to PI checkpoint directory (e.g., .../pretrained_model).",
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
    default=60,
    help="Extra frames to continue after success before ending episode. 0 = end immediately.",
)
parser.add_argument(
    "--video",
    type=str,
    default="logs/eval_videos",
    help="Output directory for per-episode MP4 videos. If not set, no video is recorded.",
)
parser.add_argument(
    "--plot-dir",
    type=str,
    default="logs/eval_plots",
    help="Output directory for state/action plots. Default: logs/eval_plots.",
)
parser.add_argument(
    "--deterministic",
    action="store_true",
    help="Enable PhysX enhanced determinism for reproducible eval (at slight perf cost).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

# Random seed for reproducibility (edit if needed)
EVAL_SEED = 42

import gymnasium as gym
from lerobot.policies.pi0 import PI0Policy
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

# State key(s) to extract from observation.
STATE_KEY_ORDER = [
    "joint_pos_7d",         # V1 base: current joint positions
    # "joint_pos_target_7d",
]

def _build_relative_mask(exclude_joints: list[str], action_names: list[str] | None, action_dim: int) -> list[bool]:
    """Build a boolean mask: True = use relative action, False = absolute (excluded)."""
    if not exclude_joints or action_names is None:
        return [True] * action_dim
    exclude_tokens = [str(name).lower() for name in exclude_joints if name]
    if not exclude_tokens:
        return [True] * action_dim
    mask = []
    for name in action_names[:action_dim]:
        action_name = str(name).lower()
        is_excluded = any(token == action_name or token in action_name for token in exclude_tokens)
        mask.append(not is_excluded)
    if len(mask) < action_dim:
        mask.extend([True] * (action_dim - len(mask)))
    return mask


def to_absolute_actions(actions: torch.Tensor, state: torch.Tensor, mask: list[bool]) -> torch.Tensor:
    """Convert relative actions back to absolute: absolute = relative + state (for masked dims).

    Args:
        actions: (*, action_dim).
        state: (*, state_dim).
        mask: Which dims are relative (True = relative → add state offset).
    """
    mask_t = torch.tensor(mask, dtype=actions.dtype, device=actions.device)
    dims = mask_t.shape[0]
    if state.device != actions.device or state.dtype != actions.dtype:
        state = state.to(device=actions.device, dtype=actions.dtype)
    state_offset = state[..., :dims] * mask_t
    actions = actions.clone()
    actions[..., :dims] += state_offset
    return actions


# ---------------------------------------------------------------------------
# Policy loading
# ---------------------------------------------------------------------------
def _patch_transformers_siglip_check():
    """Monkey-patch the missing transformers siglip check for PI0 inference.

    The installed transformers package lacks the ``check`` module that lerobot's
    PI0/PI05 expects (it comes from a custom ``transformers_replace`` package).
    Since the check only verifies that training-time patches are installed and
    does not affect inference, we inject a dummy pass-through.
    """
    import transformers.models.siglip as siglip_module
    import types
    if not hasattr(siglip_module, "check"):
        dummy = types.ModuleType("check")
        dummy.check_whether_transformers_replace_is_installed_correctly = lambda: True
        siglip_module.check = dummy


def load_unnormalizer(checkpoint_dir: str, device: torch.device):
    """Load the action unnormalizer from the checkpoint's postprocessor weights."""
    import safetensors.torch as sft
    ckpt_path = Path(checkpoint_dir)
    # The postprocessor unnormalizer is in policy_postprocessor_step_0_unnormalizer_processor.safetensors
    postproc_path = ckpt_path / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    if postproc_path.exists():
        return sft.load_file(str(postproc_path), device=str(device))
    return None


def unnormalize_action(action: torch.Tensor, unnorm_params: dict | None) -> torch.Tensor:
    """Apply QUANTILES unnormalization: action * std + mean."""
    if unnorm_params is None:
        return action
    # The safetensors contains quantile stats; for QUANTILES it's typically
    # action = action * action_std + action_mean
    mean = unnorm_params.get("action_mean") or unnorm_params.get("mean")
    std = unnorm_params.get("action_std") or unnorm_params.get("std")
    if mean is not None and std is not None:
        action = action * std.to(action.device) + mean.to(action.device)
    return action


def load_policy(checkpoint_dir: str, device: torch.device):
    """Load a PI0.5 policy from the given checkpoint directory."""
    import tempfile, shutil, os

    ckpt_path = Path(checkpoint_dir)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Monkey-patch the transformers siglip check — lerobot requires a patched
    # transformers that isn't available on PyPI. The check itself is not needed
    # for inference; it only validates that the right patches are installed.
    _patch_transformers_siglip_check()

    # Clean training-only fields from config.json that aren't recognized by
    # the installed lerobot's PI05Config (version mismatch between training and deployment).
    config_path = ckpt_path / "config.json"
    with open(config_path) as f:
        raw_config = json.load(f)

    _PI_INVALID_KEYS = {
        "pretrained_revision", "use_relative_actions", "relative_exclude_joints",
        "action_feature_names", "use_amp", "use_peft", "push_to_hub", "repo_id",
        "private", "tags", "license",
    }
    cleaned_config = {k: v for k, v in raw_config.items() if k not in _PI_INVALID_KEYS}

    # Write cleaned config to a temp directory alongside the model weights
    tmp_dir = tempfile.mkdtemp(prefix="pi_cleaned_")
    try:
        with open(os.path.join(tmp_dir, "config.json"), "w") as f:
            json.dump(cleaned_config, f, indent=2)
        # Copy model + processor weights to temp dir
        for fname in ckpt_path.glob("*.safetensors"):
            shutil.copy(str(fname), os.path.join(tmp_dir, fname.name))

        policy = PI0Policy.from_pretrained(tmp_dir)
        policy.to(device)
        policy.eval()
        logger.info(f"PI0.5 policy loaded, device={device}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Pre/post processors from lerobot.
    # The checkpoint was trained with relative_actions_processor / absolute_actions_processor
    # steps which are only in bleeding-edge lerobot. We disable them and handle the
    # relative→absolute conversion manually via `to_absolute_actions()`.
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=checkpoint_dir,
        preprocessor_overrides={"relative_actions_processor": {"enabled": False}},
        postprocessor_overrides={"absolute_actions_processor": {"enabled": False}},
    )
    logger.info("Pre/post processors created")

    # Build the relative→absolute mask from config
    exclude_joints = policy.config.__dict__.get("relative_exclude_joints", ["gripper"])
    action_names = policy.config.__dict__.get("action_feature_names", None)
    expected_action_dim = policy.config.output_features["action"].shape[0]
    _rel_mask = _build_relative_mask(exclude_joints, action_names, expected_action_dim)
    logger.info(f"Relative action mask: {_rel_mask} (exclude={exclude_joints})")

    # Extract expected dims from config
    state_ft = policy.config.input_features.get(STATE_KEY)
    expected_state_dim = state_ft.shape[0] if state_ft is not None else 7
    action_ft = policy.config.output_features.get("action")
    expected_action_dim = action_ft.shape[0] if action_ft is not None else 7

    return policy, preprocessor, postprocessor, expected_state_dim, expected_action_dim, _rel_mask


# ---------------------------------------------------------------------------
# Environment creation (same as ACT eval)
# ---------------------------------------------------------------------------
def create_env(task_name: str, expected_action_dim: int):
    """Create evaluation environment."""
    env_cfg = parse_env_cfg(task_name=task_name, device=args_cli.device, num_envs=1)

    if args_cli.deterministic:
        env_cfg.sim.physx.enable_enhanced_determinism = True
        logger.info("PhysX enhanced determinism enabled")

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

    if args_cli.post_success_delay > 0 and "success" in env.termination_manager.active_terms:
        _install_delayed_success(env, args_cli.post_success_delay)

    return env


# ---------------------------------------------------------------------------
# Build state tensor from env observation
# ---------------------------------------------------------------------------
def build_state_tensor(obs_group: dict, expected_dim: int, device: torch.device) -> torch.Tensor:
    """Extract the state tensor from observation by matching STATE_KEY_ORDER."""
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
# Build image batch — PI0.5 version
# ---------------------------------------------------------------------------
# PI0.5 uses IDENTITY normalization for VISUAL features, meaning the preprocessor
# passes images through as-is. The actual normalization (resize to 224x224,
# ImageNet mean/std) is done by the vision encoder's image processor internally.
# We still simulate MP4 roundtrip to match training data pipeline.

def _simulate_mp4_roundtrip(img_uint8: np.ndarray) -> np.ndarray:
    """Encode a single uint8 RGB frame as MP4 then decode back."""
    import subprocess, tempfile, os, cv2
    h, w = img_uint8.shape[:2]
    fd, tmppath = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
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
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(tmppath, fourcc, 30.0, (w, h))
            writer.write(cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))
            writer.release()
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


def build_image_batch(obs_group: dict, device: torch.device):
    """Build image batch for PI0.5 policy.

    PI0.5 preprocessor handles resize + normalization internally.
    We just extract uint8 images, simulate MP4 roundtrip, and convert to float32
    in [0, 255] range (IDENTITY normalization mapping).
    """
    batch = {}
    for obs_key, feat_key in CAM_KEY_MAP.items():
        val = obs_group.get(obs_key)
        if val is not None and isinstance(val, torch.Tensor):
            img_u8 = val.squeeze(0).cpu().numpy().astype(np.uint8)
            img_u8 = _simulate_mp4_roundtrip(img_u8)
            # PI0.5: keep in [0, 255] float — preprocessor handles normalization
            img = torch.from_numpy(img_u8).float()
            img = img.unsqueeze(0)  # (1, H, W, 3)
            if img.ndim == 4 and img.shape[-1] == 3:
                img = img.permute(0, 3, 1, 2)  # (1, 3, H, W)
            img = img.to(device)
            batch[feat_key] = img
        else:
            batch[feat_key] = torch.zeros(1, 3, 720, 1280, device=device)
    return batch


# ---------------------------------------------------------------------------
# Success check / delayed reset (same as ACT eval)
# ---------------------------------------------------------------------------
def _install_delayed_success(env, delay_frames: int) -> None:
    """Wrap env's success term so it delays returning True by `delay_frames`."""
    mgr = env.termination_manager
    term_idx = mgr._term_name_to_term_idx["success"]
    original_fn = mgr._term_cfgs[term_idx].func

    env._success_detected = False
    env._success_delay_left = 0

    def _delayed(env, **kwargs):
        raw = original_fn(env, **kwargs)
        if raw.any() and not env._success_detected:
            env._success_detected = True
            env._success_delay_left = delay_frames
            return torch.zeros_like(raw)
        if env._success_detected and env._success_delay_left > 0:
            env._success_delay_left -= 1
            return torch.zeros_like(raw)
        return raw

    mgr._term_cfgs[term_idx].func = _delayed


# ---------------------------------------------------------------------------
# Video writing
# ---------------------------------------------------------------------------
def check_success(env, obs: dict) -> bool:
    """Check if episode succeeded."""
    try:
        mgr = getattr(env, "termination_manager", None)
        if mgr is not None and "success" in mgr.active_terms:
            return bool(mgr.get_term("success")[0].item())
    except Exception:
        pass
    return False


def compute_sub_signals(env) -> dict:
    """Compute per-component sub-signals by directly querying scene objects."""
    result = {"a_into_c": False, "b_into_c": False, "gripper_open": False, "success": False}
    try:
        scene = env.scene
        object_a = scene["object_1"]
        object_b = scene["mug"]
        object_c = scene["box"]
        robot = scene["robot"]

        xy_threshold = 0.07
        height_threshold = 0.1

        pos_diff_a = object_a.data.root_pos_w - object_c.data.root_pos_w
        xy_dist_a = torch.linalg.vector_norm(pos_diff_a[:, :2], dim=1)
        height_dist_a = pos_diff_a[:, 2]
        a_into_c = torch.logical_and(xy_dist_a < xy_threshold, height_dist_a < height_threshold)

        pos_diff_b = object_b.data.root_pos_w - object_c.data.root_pos_w
        xy_dist_b = torch.linalg.vector_norm(pos_diff_b[:, :2], dim=1)
        height_dist_b = pos_diff_b[:, 2]
        b_into_c = torch.logical_and(xy_dist_b < xy_threshold, height_dist_b < height_threshold)

        if hasattr(env.cfg, "gripper_joint_names"):
            gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
            gripper_pos = robot.data.joint_pos[:, gripper_joint_ids]
            open_targets = torch.tensor(env.cfg.gripper_open_vals, device=gripper_pos.device).unsqueeze(0)
            gripper_open = torch.all(torch.abs(gripper_pos - open_targets) < env.cfg.gripper_threshold, dim=1)
        else:
            gripper_open = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

        result["a_into_c"] = bool(a_into_c[0].item())
        result["b_into_c"] = bool(b_into_c[0].item())
        result["gripper_open"] = bool(gripper_open[0].item())
        result["success"] = result["a_into_c"] and result["b_into_c"] and result["gripper_open"]
    except Exception:
        pass
    return result


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

    # 0. Register missing processor steps (relative_actions/absolute_actions)
    _register_missing_processor_steps()

    # 1. Load policy
    policy, preprocessor, postprocessor, expected_state_dim, expected_action_dim, rel_mask = load_policy(
        args_cli.checkpoint_dir, device
    )
    logger.info(f"Expected dims from config: state={expected_state_dim}, action={expected_action_dim}")

    # 2. Create env
    env = create_env(args_cli.task, expected_action_dim)
    env.seed(EVAL_SEED)

    # 3. Video & plot setup
    video_dir = None
    if args_cli.video:
        ckpt_parts = Path(args_cli.checkpoint_dir).resolve().parts
        try:
            idx = ckpt_parts.index("checkpoints")
            policy_name = ckpt_parts[idx - 1]
            checkpoint_name = ckpt_parts[idx + 1]
        except (ValueError, IndexError):
            policy_name = Path(args_cli.checkpoint_dir).parent.name
            checkpoint_name = Path(args_cli.checkpoint_dir).name
        video_dir = Path(args_cli.video) / f"{policy_name}-{checkpoint_name}"
        video_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Video output: {video_dir}")

    plot_dir = Path(args_cli.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Plot output: {plot_dir}")

    success_count = 0
    total_steps = 0
    joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]

    sub_signal_names = ["a_into_c", "b_into_c", "gripper_open", "success"]
    sub_signal_counts = {k: 0 for k in sub_signal_names}

    for ep in range(args_cli.num_episodes):
        logger.info(f"{'='*50}")
        logger.info(f"Episode {ep + 1}/{args_cli.num_episodes}")
        print(f"\n=== Episode {ep + 1}/{args_cli.num_episodes} ===", flush=True)

        obs, _ = env.reset()

        # Warmup: flush rendering pipeline to avoid visual ghosting
        WARMUP_STEPS = 5
        for _ in range(WARMUP_STEPS):
            obs, _, _, _, _ = env.step(
                torch.zeros(env.action_space.shape, device=env.device)
            )

        episode_steps = 0
        episode_frames = [] if video_dir else None
        ep_states: list[np.ndarray] = []
        ep_actions: list[np.ndarray] = []

        # One-time debug
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

        # PI0.5 chunk execution
        CHUNK_EXEC_STEPS = 50  # Execute full chunk before re-inferring

        action_chunk = None
        chunk_pos = 0
        chunk_len = 0

        post_success_delay = 0
        episode_success = False
        env._success_detected = False
        env._success_delay_left = 0

        while episode_steps < args_cli.max_steps or post_success_delay > 0:
            obs_group = obs.get("policy", obs)

            # --- Capture frame ---
            if episode_frames is not None:
                front = obs_group.get("table_cam")
                if front is not None and isinstance(front, torch.Tensor):
                    f = front.squeeze(0).cpu().numpy().astype(np.uint8)
                    episode_frames.append(f)

            # --- Build policy input ---
            batch = build_image_batch(obs_group, device)

            # One-time check
            if ep == 0 and episode_steps == 0:
                for k in ["observation.images.front", "observation.images.wrist"]:
                    img = batch.get(k)
                    if img is not None and isinstance(img, torch.Tensor):
                        print(f"[DEBUG] raw {k}: shape={list(img.shape)}, min={img.min().item():.1f}, "
                              f"max={img.max().item():.1f}, mean={img.mean().item():.1f}", flush=True)
                        if img.max().item() == 0.0:
                            print(f"[WARN] ⚠️  {k} is ALL ZEROS — cameras may not be rendering! Use --enable_cameras.", flush=True)

            raw_state = build_state_tensor(obs_group, expected_state_dim, device)
            batch[STATE_KEY] = raw_state

            # Preprocess (handles resize to 224x224, normalization, etc.)
            batch = preprocessor(batch)

            if ep == 0 and episode_steps == 0:
                ns = batch.get(STATE_KEY)
                if ns is not None:
                    print(f"[DEBUG] normalized state: shape={list(ns.shape)}, values={ns.squeeze().tolist()[:10]}", flush=True)
                for imgk in policy.config.image_features:
                    img = batch.get(imgk)
                    if img is not None and isinstance(img, torch.Tensor):
                        print(f"[DEBUG] preprocessed {imgk}: shape={list(img.shape)}, min={img.min().item():.4f}, "
                              f"max={img.max().item():.4f}, mean={img.mean().item():.4f}", flush=True)

            # --- Re-infer when chunk exhausted ---
            need_reinfer = action_chunk is None or chunk_pos >= chunk_len
            if CHUNK_EXEC_STEPS is not None and chunk_pos >= CHUNK_EXEC_STEPS:
                need_reinfer = True
            if need_reinfer:
                with torch.inference_mode():
                    action_chunk = policy.predict_action_chunk(batch)  # (1, n_action_steps, 7)
                action_chunk = action_chunk.cpu()
                chunk_len = action_chunk.shape[1]
                chunk_pos = 0

            # Take next action from chunk
            action = action_chunk[:, chunk_pos, :].clone()

            # Postprocess: unnormalize (normalizer_processor handles this)
            action = postprocessor(action)
            action = action.squeeze(0)  # (action_dim,)

            # Convert relative actions to absolute using current state.
            # PI0.5 was trained with use_relative_actions=True (joint1-6 are delta,
            # gripper is absolute). The postprocessor's absolute_actions_processor was
            # disabled, so we do this manually.
            if any(rel_mask):
                action = to_absolute_actions(action, raw_state[0], rel_mask)

            action = action.cpu().numpy()
            chunk_pos += 1

            # Record state & action
            state_np = raw_state.squeeze(0).cpu().numpy()
            ep_states.append(state_np.copy())
            ep_actions.append(action.copy())

            # Step env
            env_action = torch.from_numpy(action).reshape(env.action_space.shape).to(env.device)
            obs, reward, terminated, truncated, info = env.step(env_action)

            if ep == 0 and episode_steps == 0:
                robot = env.scene["robot"]
                jt = robot.data.joint_pos_target[0].cpu().numpy()
                jp = robot.data.joint_pos[0].cpu().numpy()
                print(f"[DEBUG] after step 0: joint_pos={np.array2string(jp, precision=3, suppress_small=True)}", flush=True)
                print(f"[DEBUG] after step 0: joint_target={np.array2string(jt, precision=3, suppress_small=True)}", flush=True)
                print(f"[DEBUG] after step 0: model_action={np.array2string(action, precision=3, suppress_small=True)}", flush=True)
                print(f"[DEBUG] chunk_len={chunk_len}", flush=True)

            episode_steps += 1
            total_steps += 1

            if episode_steps % 100 == 0 or episode_steps == 1:
                print(f"[Ep {ep + 1}/{args_cli.num_episodes}] Step {episode_steps}/{args_cli.max_steps}", flush=True)

            # Check termination
            if args_cli.post_success_delay > 0:
                if post_success_delay == 0 and env._success_detected:
                    episode_success = True
                    post_success_delay = args_cli.post_success_delay
                    print(f"[INFO] Success! Continuing {post_success_delay} frames naturally...", flush=True)

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

        sub = compute_sub_signals(env)
        for k in sub_signal_names:
            if sub[k]:
                sub_signal_counts[k] += 1

        sub_parts = []
        for k in ["a_into_c", "b_into_c", "gripper_open"]:
            mark = "✓" if sub[k] else "✗"
            sub_parts.append(f"{k}={mark}")
        sub_status = "  ".join(sub_parts)

        if episode_success:
            print(f"  ✅ Episode {ep + 1} SUCCESS  [{sub_status}] ({episode_steps} steps)", flush=True)
        else:
            print(f"  ❌ Episode {ep + 1} FAILED  [{sub_status}] ({episode_steps} steps)", flush=True)

        # --- Plot state/action ---
        if ep_states and ep_actions:
            states_arr = np.array(ep_states)
            actions_arr = np.array(ep_actions)
            steps = np.arange(len(ep_states))

            fig, axes = plt.subplots(4, 2, figsize=(16, 14))
            axes = axes.flatten()
            for j in range(7):
                ax = axes[j]
                ax.plot(steps, actions_arr[:, j], "-", linewidth=1.5, label="action", color="C0")
                ax.plot(steps, states_arr[:, j], "--", linewidth=1.5, label="state", color="C1")
                ax.set_title(joint_names[j])
                ax.set_xlabel("step")
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=7)
            axes[7].set_visible(False)

            status = "SUCCESS" if episode_success else "FAILED"
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
    failed_count = args_cli.num_episodes - success_count
    print(f"\n{'='*50}", flush=True)
    print(f"Done: {success_count}/{args_cli.num_episodes} SUCCESS ({100 * success_count / max(1, args_cli.num_episodes):.1f}%)", flush=True)
    print(f"      {failed_count}/{args_cli.num_episodes} FAILED (timeout or early termination)", flush=True)
    print(f"Total steps: {total_steps}", flush=True)
    print(f"\n--- Sub-signal breakdown (end-of-episode) ---", flush=True)
    for k in ["a_into_c", "b_into_c", "gripper_open"]:
        pct = 100 * sub_signal_counts[k] / max(1, args_cli.num_episodes)
        print(f"  {k}: {sub_signal_counts[k]}/{args_cli.num_episodes} ({pct:.1f}%)", flush=True)
    print(f"Plots saved to: {plot_dir}", flush=True)
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
