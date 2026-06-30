#!/usr/bin/env python3
"""
使用训练好的 ACT 策略在 Visuomotor 环境中评估。

用法 (容器内):
    ./isaaclab.sh -p scripts/lerobot/eval_act_piper_grab.py \\
        --checkpoint /workspace/isaaclab/datasets/lerobot/piper_grab_V1_D3_checkpoints/checkpoint_step_100000 \\
        --num_episodes 10 \\
        --headless --enable_cameras --device cuda:0

    要渲染画面:
    ./isaaclab.sh -p scripts/lerobot/eval_act_piper_grab.py \\
        --checkpoint /workspace/isaaclab/datasets/lerobot/piper_grab_V1_D3_checkpoints/checkpoint_step_100000 \\
        --num_episodes 10 \\
        --device cuda:0
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.v2 as T

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate ACT policy on Piper Grab Visuomotor V1 task.")
parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="Path to ACT policy checkpoint directory (e.g., .../checkpoint_step_100000).",
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

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg, BinaryJointPositionActionCfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ACT 相关常量 (与训练时一致)
# ---------------------------------------------------------------------------
IMG_RESIZE = (224, 224)
CHUNK_SIZE = 50      # 从 checkpoint config 读取，此处作为 fallback
DEVICE_STR = "cpu" if args_cli.policy_cpu else args_cli.device

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
    import json

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
# 环境辅助
# ---------------------------------------------------------------------------
def create_env(task_name: str = "Isaac-Piper-Grab-IK-Rel-Visuomotor-Mimic-v1"):
    """创建评估环境，将 action 改为 joint position 模式以匹配 policy 输出。"""
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    env_cfg = parse_env_cfg(
        task_name=task_name,
        device=args_cli.device,
        num_envs=1,
    )

    # --- 关闭数据生成相关选项 ---
    env_cfg.datagen_config.generation_num_trials = 1

    # --- 保留随机化（与数据生成一致），仅调整观察 ---
    env_cfg.observations.policy.concatenate_terms = False

    # --- 替换 action: Cartesian IK → 直接 joint position ---
    # policy 输出 8D: [joint1..joint6, gripper_joint7, gripper_joint8]
    env_cfg.actions.arm_action = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint[1-6]"],
        scale=1.0,
    )
    # 用绝对值控制夹爪 (joint7, joint8)
    env_cfg.actions.gripper_action = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint7"],
        open_command_expr={"joint7": 0.05},
        close_command_expr={"joint7": -0.05},
    )
    # 保留 gripper 判断用的配置
    env_cfg.gripper_joint_names = ["joint7", "joint8"]
    env_cfg.gripper_open_vals = [0.05, -0.05]
    env_cfg.gripper_threshold = 0.01

    # --- 关闭 recorder ---
    env_cfg.recorders = None

    env = gym.make(task_name, cfg=env_cfg).unwrapped

    # 注册 env 名称以便后续 parse_env_cfg 缓存识别
    if not hasattr(env.cfg, "env_name") or not env.cfg.env_name:
        env.cfg.env_name = task_name

    logger.info(f"Environment created: {task_name}")
    logger.info(f"  Action space: {env.action_space}")
    logger.info(f"  Obs space keys: {list(env.observation_space.keys()) if hasattr(env.observation_space, 'keys') else env.observation_space}")
    return env


def rescale_gripper_action(gripper_val: float) -> float:
    """将 policy 输出的 gripper joint 值转换为 BinaryJointPositionAction 接受的标量。

    policy 输出 gripper joint7 的绝对值（~0.05=开, ~-0.05=关）。
    BinaryJointPositionAction 接受正数为开、负数为关。
    """
    # 直接返回原始值，BinaryJointPositionAction 按正负判断开关
    return float(gripper_val)


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

    # 2. 创建环境
    env = create_env()
    image_transforms = build_image_transforms()

    # 3. 确定 camera key mapping (env obs key → lerobot feature key)
    cam_key_map = {
        "table_cam": "observation.images.front",
        "wrist_cam": "observation.images.wrist",
    }
    state_key = "observation.state"
    policy_obs_group = "policy"  # 观察所属的 group 名

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
            # 从观测 dict 中提取 group
            obs_group = obs.get(policy_obs_group, obs)

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
            state_key_order = [
                "actions", "joint_pos", "joint_vel", "object",
                "eef_pos", "eef_quat", "gripper_pos",
                "object_1_positions", "object_1_orientations",
                "box_positions", "box_orientations",
                "mug_positions", "mug_orientations",
            ]
            state_parts = []
            for key in state_key_order:
                val = obs_group.get(key)
                if val is not None and isinstance(val, torch.Tensor):
                    v = val.float().reshape(-1)
                    if v.numel() == 0:
                        continue
                    state_parts.append(v)
            state_tensor = torch.cat(state_parts).unsqueeze(0).to(device)  # (1, 63)
            batch[state_key] = state_tensor

            # Images: resize + ImageNet 标准化 (与训练时一致), 再由 preprocessor 归一化
            for obs_key, feat_key in cam_key_map.items():
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
                    # 应用与训练时一致的 image transforms
                    img = image_transforms(img)
                    batch[feat_key] = img
                else:
                    batch[feat_key] = torch.zeros(1, 3, *IMG_RESIZE, device=device)

            # --- Preprocess & Infer ---
            batch = preprocessor(batch)

            # ACT model 要求将图像汇聚到 observation.images 列表
            # (ACTPolicy.forward() 会做这步，但我们直接用 policy.model)
            if policy.config.image_features:
                batch = dict(batch)  # shallow copy
                batch["observation.images"] = [batch[k] for k in policy.config.image_features]

            with torch.inference_mode():
                actions_hat, _ = policy.model(batch)  # (1, chunk_size, 8)

            # 取第一帧 action，后处理 (postprocessor 接收 tensor, 需要 CPU)
            action = actions_hat[:, 0, :].cpu()  # (1, 8)
            action = postprocessor(action).squeeze(0).numpy()  # (8,)

            # --- 映射到 env action ---
            # action[0:6] = joint1..joint6 位置
            # action[6:8] = joint7, joint8 位置
            arm_action = action[0:6]
            gripper_raw = action[6]  # joint7 值 (~0.05=开, ~-0.05=关)

            env_action = np.zeros(np.prod(env.action_space.shape), dtype=np.float32)
            env_action[0:6] = arm_action
            env_action[6] = rescale_gripper_action(gripper_raw)
            # 如果 action space 是 8D，第 8 维直接用
            if env.action_space.shape[0] >= 8:
                env_action[7] = float(action[7])

            # --- Step ---
            obs, reward, terminated, truncated, info = env.step(
                torch.from_numpy(env_action).reshape(env.action_space.shape).to(env.device)
            )

            episode_done = terminated.item() if hasattr(terminated, 'item') else bool(terminated)

            if episode_done:
                # 检查 success termination 是否触发
                is_success = False
                if hasattr(env, "termination_manager") and env.termination_manager is not None:
                    success_buf = env.termination_manager._term_buf.get("success")
                    if success_buf is not None:
                        is_success = bool(success_buf[0].item()) if hasattr(success_buf, '__getitem__') else False
                if is_success:
                    success_count += 1
                    logger.info(f"  ✅ Episode {episode + 1} SUCCESS ({episode_steps} steps)")
                else:
                    logger.info(f"  ❌ Episode {episode + 1} FAILED ({episode_steps} steps)")

            episode_steps += 1
            total_steps += 1

        # --- 写入视频 ---
        if episode_frames is not None and len(episode_frames) > 0:
            import torchvision.io as tio
            frames_tensor = torch.from_numpy(np.stack(episode_frames))  # (T, H, W, C)
            video_path = video_dir / f"episode_{episode:03d}.mp4"
            tio.write_video(str(video_path), frames_tensor, fps=30)
            logger.info(f"  🎥 Video saved: {video_path}")

        if not episode_done:
            logger.info(f"  ⏱️  Episode {episode + 1} TIMEOUT ({episode_steps} steps)")

    # 4. 打印汇总
    logger.info(f"{'='*50}")
    logger.info(f"Evaluation complete: {success_count}/{args_cli.num_episodes} successful")
    logger.info(f"Success rate: {success_count / args_cli.num_episodes * 100:.1f}%")
    logger.info(f"Total steps: {total_steps}")

    env.close()


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
