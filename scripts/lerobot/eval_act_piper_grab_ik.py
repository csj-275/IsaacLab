#!/usr/bin/env python3
"""
使用训练好的 ACT 策略在 Visuomotor 环境中评估（IK action space）。

与 eval_act_piper_grab.py 的区别：
  - 保留原始的 DifferentialInverseKinematicsActionCfg（IK delta pose），
    而非替换为 JointPositionActionCfg
  - Policy 输出的 8D action 直接传给 env.step()，由 IK 控制器解释为
    [dx, dy, dz, droll, dpitch, dyaw, gripper_joint7, gripper_joint8]

用法 (容器内):
    ./isaaclab.sh -p scripts/lerobot/eval_act_piper_grab_ik.py \
        --checkpoint /workspace/isaaclab/datasets/lerobot/piper_grab_V1_D3_checkpoints/checkpoint_step_100000 \
        --num_episodes 10 \
        --headless --enable_cameras --device cuda:0

    要渲染画面:
    ./isaaclab.sh -p scripts/lerobot/eval_act_piper_grab_ik.py \
        --checkpoint /workspace/isaaclab/datasets/lerobot/piper_grab_V1_D3_checkpoints/checkpoint_step_100000 \
        --num_episodes 10 \
        --device cuda:0
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.v2 as T

# lerobot 已通过 pip 安装，无需额外路径

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate ACT policy on Piper Grab Visuomotor V1 task (IK action space).")
parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="Path to ACT policy checkpoint directory (e.g., .../checkpoint_step_100000).",
)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Piper-Grab-IK-Rel-Visuomotor-v1",
    help="Isaac Lab task name (default: Isaac-Piper-Grab-IK-Rel-Visuomotor-v1).",
)
parser.add_argument("--num_episodes", type=int, default=10, help="Number of evaluation episodes.")
parser.add_argument("--max_steps_per_episode", type=int, default=500, help="Max steps per episode.")
parser.add_argument("--policy-cpu", action="store_true", help="Load policy on CPU (for debugging).")
parser.add_argument(
    "--record-video-to",
    type=str,
    default=None,
    help="Output directory for video recording. If set, front camera frames are saved as MP4 per episode.",
)
parser.add_argument("--no-close", action="store_true", help="Keep window open after evaluation finishes.")
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

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量 (与训练时一致)
# ---------------------------------------------------------------------------
IMG_RESIZE = (224, 224)
CHUNK_SIZE = 50       # 从 checkpoint config 读取，此处作为 fallback
DEVICE_STR = "cpu" if args_cli.policy_cpu else args_cli.device

# State 维度顺序（必须与训练数据完全一致，共 63D）
STATE_KEY_ORDER = [
    "actions",                  #  8D — 上一帧 action
    "joint_pos",                #  7D — joints 1-6 + joint7
    "joint_vel",                #  7D — joint velocities
    "object",                   # 10D — cube pos(3) + quat(4) + gripper-to-cube(3)
    "eef_pos",                  #  3D — end-effector position
    "eef_quat",                 #  4D — end-effector orientation (w,x,y,z)
    "gripper_pos",              #  2D — joint7, -joint8
    "object_1_positions",       #  3D — cube pos in robot base frame
    "object_1_orientations",    #  4D — cube quat in robot base frame
    "box_positions",            #  3D — box pos in robot base frame
    "box_orientations",         #  4D — box quat in robot base frame
    "mug_positions",            #  3D — mug pos in robot base frame
    "mug_orientations",         #  4D — mug quat in robot base frame
]
# 总计: 8+7+7+10+3+4+2+3+4+3+4+3+4 = 63

# Camera key mapping (env obs key → lerobot feature key)
CAM_KEY_MAP = {
    "table_cam": "observation.images.front",
    "wrist_cam": "observation.images.wrist",
}

# State feature key in LeRobot format
STATE_KEY = "observation.state"


# ---------------------------------------------------------------------------
# 图像预处理 (与训练时完全一致: resize + ImageNet 标准化)
# ---------------------------------------------------------------------------
def build_image_transforms():
    return T.Compose([
        T.Resize(IMG_RESIZE, antialias=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------------------------
# 从 checkpoint 恢复信息
# ---------------------------------------------------------------------------
def load_policy(checkpoint_dir: str, device: torch.device):
    """从 checkpoint 加载 ACT 策略和前后处理器。"""
    ckpt_path = Path(checkpoint_dir)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # 手动加载 config.json，去掉 draccus 不认识的字段
    config_path = ckpt_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        raw_config = json.load(f)

    # 保留 features 信息（后面重建 PolicyFeature 对象）
    input_features_raw = raw_config.pop("input_features", {})
    output_features_raw = raw_config.pop("output_features", {})
    raw_config.pop("type", None)  # draccus 不认这个字段

    cfg = ACTConfig(**raw_config)
    # 重建 input/output features
    cfg.input_features = {
        k: PolicyFeature(type=FeatureType(v["type"]), shape=tuple(v["shape"]))
        for k, v in input_features_raw.items()
    }
    cfg.output_features = {
        k: PolicyFeature(type=FeatureType(v["type"]), shape=tuple(v["shape"]))
        for k, v in output_features_raw.items()
    }
    logger.info(f"Loaded config from {config_path}")
    logger.info(f"  Input features: {list(cfg.input_features.keys())}")
    logger.info(f"  Output features: {list(cfg.output_features.keys())}")

    # 恢复 policy 权重 (使用 safetensors)
    import safetensors.torch as sft
    state_dict = sft.load_file(str(ckpt_path / "model.safetensors"), device=str(device))
    policy = ACTPolicy(cfg)
    policy.load_state_dict(state_dict)
    policy.to(device)
    policy.eval()
    logger.info(f"Policy loaded from {ckpt_path}, device={device}")

    # 前后处理器
    preprocessor, postprocessor = make_pre_post_processors(cfg, pretrained_path=str(ckpt_path))
    logger.info("Pre/post processors loaded")

    chunk_size = getattr(cfg, "chunk_size", CHUNK_SIZE)
    n_action_steps = getattr(cfg, "n_action_steps", CHUNK_SIZE)
    return policy, preprocessor, postprocessor, chunk_size, n_action_steps


# ---------------------------------------------------------------------------
# 环境创建 (保留 IK arm action，覆盖 gripper 为 2D joint position)
# ---------------------------------------------------------------------------
def create_env(task_name: str = "Isaac-Piper-Grab-IK-Rel-Visuomotor-v1"):
    """创建评估环境，对齐 8D action space。

    训练数据的 action 是 8D: [dx, dy, dz, droll, dpitch, dyaw, joint7, joint8]。
    但默认配置的 gripper 使用 MimicBinaryJointPositionActionCfg（1D scalar），
    导致 action space 只有 7D（6 IK + 1 gripper scalar）。

    这里将 gripper 覆盖为 JointPositionActionCfg(joint7, joint8) → 2D，
    总 action space = 6 (IK) + 2 (gripper) = 8D，与 policy 输出对齐。
    """
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    env_cfg = parse_env_cfg(
        task_name=task_name,
        device=args_cli.device,
        num_envs=1,
    )

    # --- 覆盖 gripper action: 2D joint position（匹配训练数据的 8D action）---
    env_cfg.actions.gripper_action = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint7", "joint8"],
        scale=1.0,
    )
    # 保留 gripper 判断用的配置（与默认一致）
    env_cfg.gripper_joint_names = ["joint7", "joint8"]
    env_cfg.gripper_open_vals = [0.05, -0.05]
    env_cfg.gripper_threshold = 0.01

    # --- 关闭数据生成相关选项 ---
    if hasattr(env_cfg, "datagen_config"):
        env_cfg.datagen_config.generation_num_trials = 1

    # --- 保留随机化（与数据生成一致），仅调整观察 ---
    env_cfg.observations.policy.concatenate_terms = False

    # --- 关闭 recorder ---
    env_cfg.recorders = None

    # --- 关闭 terminations（手动检测 success）---
    env_cfg.terminations = None

    env = gym.make(task_name, cfg=env_cfg).unwrapped

    # 注册 env 名称以便后续 parse_env_cfg 缓存识别
    if not hasattr(env.cfg, "env_name") or not env.cfg.env_name:
        env.cfg.env_name = task_name

    logger.info(f"Environment created: {task_name}")
    logger.info(f"  Action space: {env.action_space}")
    logger.info(f"  Arm: DifferentialInverseKinematicsActionCfg (6D IK delta pose)")
    logger.info(f"  Gripper: JointPositionActionCfg(joint7, joint8) (2D)")
    return env


# ---------------------------------------------------------------------------
# 主评估循环
# ---------------------------------------------------------------------------
def main():
    device = torch.device(DEVICE_STR)

    # 1. 加载策略
    policy, preprocessor, postprocessor, chunk_size, n_action_steps = load_policy(
        args_cli.checkpoint, device
    )
    logger.info(f"Policy: chunk_size={chunk_size}, n_action_steps={n_action_steps}")

    # 2. 创建环境（保留 IK action config）
    env = create_env(args_cli.task)
    image_transforms = build_image_transforms()

    success_count = 0
    total_steps = 0

    # 视频录制路径
    video_dir = None
    if args_cli.record_video_to:
        video_dir = Path(args_cli.record_video_to)
        video_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Video recording to: {video_dir}")

    for episode in range(args_cli.num_episodes):
        logger.info(f"{'='*50}")
        logger.info(f"Episode {episode + 1}/{args_cli.num_episodes}")

        obs, _ = env.reset()
        episode_steps = 0
        episode_done = False

        # 收集 episode 帧用于视频录制
        episode_frames = [] if video_dir else None

        while not episode_done and episode_steps < args_cli.max_steps_per_episode:
            # 从观测 dict 中提取 policy group
            obs_group = obs.get("policy", obs)

            # --- 录制视频帧 (table_cam) ---
            if episode_frames is not None:
                front_img = obs_group.get("table_cam")
                if front_img is not None and isinstance(front_img, torch.Tensor):
                    frame = front_img.squeeze(0)  # (H, W, 3)
                    frame_np = frame.cpu().numpy().astype(np.uint8)
                    episode_frames.append(frame_np)

            # --- 构建 policy 输入 ---
            batch = {}

            # State (63D): 按训练时的观察组顺序拼接
            state_parts = []
            for key in STATE_KEY_ORDER:
                val = obs_group.get(key)
                if val is not None and isinstance(val, torch.Tensor):
                    v = val.float().reshape(-1)
                    if v.numel() == 0:
                        continue
                    state_parts.append(v)
                else:
                    logger.warning(f"State key '{key}' not found in observation, using zeros")
                    # 根据已知维度填充零
                    known_dims = {
                        "actions": 8, "joint_pos": 7, "joint_vel": 7, "object": 10,
                        "eef_pos": 3, "eef_quat": 4, "gripper_pos": 2,
                        "object_1_positions": 3, "object_1_orientations": 4,
                        "box_positions": 3, "box_orientations": 4,
                        "mug_positions": 3, "mug_orientations": 4,
                    }
                    dim = known_dims.get(key, 0)
                    if dim > 0:
                        state_parts.append(torch.zeros(dim))
            state_tensor = torch.cat(state_parts).unsqueeze(0).to(device)  # (1, 63)
            batch[STATE_KEY] = state_tensor

            # Images: resize + ImageNet 标准化 (与训练时一致), 再由 preprocessor 归一化
            for obs_key, feat_key in CAM_KEY_MAP.items():
                val = obs_group.get(obs_key)
                if val is not None and isinstance(val, torch.Tensor):
                    img = val.float()  # uint8 → float
                    if img.ndim == 3:
                        img = img.unsqueeze(0)  # (1, H, W, 3)
                    # NHWC → NCHW
                    if img.ndim == 4 and img.shape[-1] == 3:
                        img = img.permute(0, 3, 1, 2).float()  # (1, 3, H, W)
                    elif img.ndim == 4 and img.shape[1] == 3:
                        pass  # already (1, 3, H, W)
                    img = img.to(device)
                    # 应用与训练时一致的 image transforms (Resize + ImageNet norm)
                    img = image_transforms(img)
                    batch[feat_key] = img
                else:
                    logger.warning(f"Camera key '{obs_key}' not found, using zeros")
                    batch[feat_key] = torch.zeros(1, 3, *IMG_RESIZE, device=device)

            # --- Preprocess & Infer ---
            batch = preprocessor(batch)

            # ACT model 要求将图像汇聚到 observation.images 列表
            if policy.config.image_features:
                batch = dict(batch)  # shallow copy
                batch["observation.images"] = [batch[k] for k in policy.config.image_features]

            with torch.inference_mode():
                actions_hat, _ = policy.model(batch)  # (1, chunk_size, 8)

            # 取第一帧 action，后处理 (postprocessor 做 unnormalize)
            action = actions_hat[:, 0, :].cpu()  # (1, 8)
            action = postprocessor(action).squeeze(0).numpy()  # (8,)

            # --- 映射到 env action ---
            # IK 模式下，policy 输出的 8D action 直接对应环境的 action space:
            #   action[0:3] = delta position (dx, dy, dz)
            #   action[3:6] = delta rotation axis-angle (droll, dpitch, dyaw)
            #   action[6:8] = gripper joint7, joint8 位置
            env_action = torch.from_numpy(action).reshape(env.action_space.shape).to(env.device)

            # --- Step ---
            obs, reward, terminated, truncated, info = env.step(env_action)

            episode_done = terminated.item() if hasattr(terminated, 'item') else bool(terminated)
            episode_done = episode_done or (truncated.item() if hasattr(truncated, 'item') else bool(truncated))

            if episode_done:
                break

            episode_steps += 1
            total_steps += 1

        # --- 检查 success（每 episode 结束时检查，无论 timeout 还是 early termination）---
        is_success = _check_success(env, obs)
        if is_success:
            success_count += 1
            logger.info(f"  ✅ Episode {episode + 1} SUCCESS ({episode_steps} steps)")
        elif episode_done:
            logger.info(f"  ❌ Episode {episode + 1} FAILED ({episode_steps} steps)")
        else:
            logger.info(f"  ⏱️  Episode {episode + 1} TIMEOUT ({episode_steps} steps)")

        # --- 写入视频 ---
        if episode_frames is not None and len(episode_frames) > 0:
            import torchvision.io as tio
            frames_tensor = torch.from_numpy(np.stack(episode_frames))  # (T, H, W, C)
            video_path = video_dir / f"episode_{episode:03d}.mp4"
            tio.write_video(str(video_path), frames_tensor, fps=30)
            logger.info(f"  🎥 Video saved: {video_path}")

    # 4. 打印汇总
    logger.info(f"{'='*50}")
    logger.info(f"Evaluation complete: {success_count}/{args_cli.num_episodes} successful")
    logger.info(f"Success rate: {success_count / args_cli.num_episodes * 100:.1f}%")
    logger.info(f"Total steps: {total_steps}")

    env.close()


def _check_success(env, obs: dict) -> bool:
    """检查 episode 是否成功。

    优先通过 subtask_terms 检测 "placed_1"（cube 放入 box），
    fallback 到 termination_manager 的 success buffer。
    """
    # 方法 1: subtask_terms
    subtask_terms = obs.get("subtask_terms", {})
    placed = subtask_terms.get("placed_1")
    if placed is not None:
        if isinstance(placed, torch.Tensor):
            return bool(placed[0].item()) if placed.numel() > 0 else False
        return bool(placed)

    # 方法 2: 检查是否因 success termination 而结束
    if hasattr(env, "termination_manager") and env.termination_manager is not None:
        success_buf = env.termination_manager._term_buf.get("success")
        if success_buf is not None:
            return bool(success_buf[0].item()) if hasattr(success_buf, '__getitem__') else False

    return False


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    if args_cli.no_close:
        logger.info("Evaluation finished. Window will close when you close the viewer, or press Ctrl+C.")
        try:
            while simulation_app.is_running():
                simulation_app.update()
        except KeyboardInterrupt:
            pass
    simulation_app.close()
