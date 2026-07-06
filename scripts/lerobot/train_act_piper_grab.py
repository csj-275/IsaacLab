#!/usr/bin/env python3
"""
使用 LeRobot ACT 算法训练策略，适配 isaaclab/datasets/simdata 中的所有数据集。

用法:
    conda activate lerobot

    # 从头训练（必须指定 --dataset-root）
    python scripts/lerobot/train_act_piper_grab.py \
        --dataset-root /home/chenshengjia/company/isaaclab/datasets/lerobot/piper_grab_v1

    # 续训
    python scripts/lerobot/train_act_piper_grab.py \
        --dataset-root /home/chenshengjia/company/isaaclab/datasets/lerobot/piper_grab_v1 \
        --resume <checkpoint_dir>

    # Docker 容器内
    python scripts/lerobot/train_act_piper_grab.py \
        --dataset-root /workspace/isaaclab/datasets/lerobot/piper_grab_v1
"""

import logging
import argparse
from pathlib import Path

import torch
import torchvision.transforms.v2 as T
from tqdm import tqdm

# --- 必须在所有 lerobot import 之前执行所有 monkey-patches ---
def _apply_patches():
    import lerobot.datasets.utils as ds_utils
    import lerobot.datasets.lerobot_dataset as lds

    # Patch 1: 修复本地数据集无需联网验证 Hub 版本
    def noop_get_safe_version(repo_id, version):
        return str(version)
    ds_utils.get_safe_version = noop_get_safe_version
    lds.get_safe_version = noop_get_safe_version

    # Patch 2: 修复 pyarrow FixedSizeList child name 'element' vs HF 'item' 不兼容
    original_load = ds_utils.load_nested_dataset

    def patched_load(pq_dir, features=None, episodes=None):
        return original_load(pq_dir, features=None, episodes=episodes)

    ds_utils.load_nested_dataset = patched_load
    lds.load_nested_dataset = patched_load  # lerobot_dataset 作 module-level import，也需 patch

    # Patch 3: 强制 video backend 为 pyav（torchcodec 需要 FFmpeg 系统库，容器内不可用）
    from lerobot.datasets import video_utils as vu
    def patched_get_backend():
        return "pyav"
    vu.get_safe_default_codec = patched_get_backend
    vu.get_default_codec = patched_get_backend
    lds.get_safe_default_codec = patched_get_backend  # lerobot_dataset 也做了 module-level import


_apply_patches()

from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# 命令行参数
# ============================================================
parser = argparse.ArgumentParser(description="Train ACT policy on a LeRobot dataset")
parser.add_argument(
    "--dataset-root",
    type=str,
    required=True,
    help="LeRobot 数据集根目录路径 (包含 meta/info.json)",
)
parser.add_argument(
    "--output-dir",
    type=str,
    default=None,
    help="训练输出目录 (默认: {dataset-root} 的父目录下创建 {name}_act_checkpoints)",
)
parser.add_argument(
    "--resume",
    type=str,
    default=None,
    help="Path to checkpoint directory to resume from",
)
args = parser.parse_args()

DATASET_ROOT = Path(args.dataset_root)
if not (DATASET_ROOT / "meta" / "info.json").exists():
    raise FileNotFoundError(f"Dataset not found: {DATASET_ROOT} (missing meta/info.json)")

DATASET_NAME = DATASET_ROOT.name
OUTPUT_DIR = Path(args.output_dir) if args.output_dir else (DATASET_ROOT.parent / f"{DATASET_NAME}_act_checkpoints")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"Dataset root: {DATASET_ROOT}")
logger.info(f"Output dir: {OUTPUT_DIR}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# 训练参数
BATCH_SIZE = 8             # 双路 1280x720 图片，ResNet 占用大量显存
NUM_EPOCHS = 50           # 根据收敛情况调整
LOG_FREQ = 10
SAVE_FREQ = 5000
LR = 1e-5                  # ACT 默认学习率
LR_BACKBONE = 1e-5

# ACT 参数
CHUNK_SIZE = 50            # 预测未来多少个 action (缩小以适配较短 episode)
N_ACTION_STEPS = 50        # 每次执行多少个 action
IMG_RESIZE = (224, 224)   # 图像 resize 到 ResNet 标准输入


def build_image_transforms():
    """图像预处理: resize + imagenet 标准化."""
    return T.Compose([
        T.Resize(IMG_RESIZE, antialias=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def main():
    # 1. 加载数据集元数据
    logger.info("=" * 60)
    logger.info("Step 1: Loading dataset metadata")
    logger.info("=" * 60)
    ds_meta = LeRobotDatasetMetadata(DATASET_NAME, root=DATASET_ROOT)
    logger.info(f"Dataset: {ds_meta.total_episodes} episodes, {ds_meta.total_frames} frames, {ds_meta.fps} FPS")
    logger.info(f"Features: {list(ds_meta.features.keys())}")

    # 2. 构建 policy features
    logger.info("=" * 60)
    logger.info("Step 2: Building policy features")
    logger.info("=" * 60)
    features = dataset_to_policy_features(ds_meta.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    logger.info(f"Input features: {list(input_features.keys())}")
    logger.info(f"Output features: {list(output_features.keys())}")

    # 根据最短 episode 自动 clamp chunk_size（放在 ACT 配置之前）
    min_ep_len = min(ds_meta.episodes["length"]) if "length" in ds_meta.episodes.column_names else CHUNK_SIZE
    actual_chunk_size = min(CHUNK_SIZE, min_ep_len)
    actual_n_action_steps = min(N_ACTION_STEPS, actual_chunk_size)
    if actual_chunk_size < CHUNK_SIZE:
        logger.warning(f"CHUNK_SIZE clamped from {CHUNK_SIZE} to {actual_chunk_size} (min episode length = {min_ep_len})")

    # 3. 创建 ACT 配置
    logger.info("=" * 60)
    logger.info("Step 3: Creating ACT config")
    logger.info("=" * 60)
    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        # ACT 核心参数
        chunk_size=actual_chunk_size,
        n_action_steps=actual_n_action_steps,
        n_obs_steps=1,
        # Transformer 架构 (减小以适配 CPU 训练)
        dim_model=256,
        n_heads=4,
        dim_feedforward=1024,
        n_encoder_layers=2,
        n_decoder_layers=1,
        # VAE
        use_vae=True,
        latent_dim=32,
        n_vae_encoder_layers=2,
        kl_weight=10.0,
        # 归一化
        normalization_mapping={
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        },
        # 训练
        dropout=0.1,
        optimizer_lr=LR,
        optimizer_weight_decay=1e-4,
        optimizer_lr_backbone=LR_BACKBONE,
        # 视觉
        vision_backbone="resnet18",
        pretrained_backbone_weights=None,  # None = 从头训练，无需下载
        replace_final_stride_with_dilation=False,
    )
    logger.info(f"Config: dim_model={cfg.dim_model}, n_heads={cfg.n_heads}, "
                f"chunk_size={cfg.chunk_size}, use_vae={cfg.use_vae}")

    # 4. 创建 / 恢复 Policy
    logger.info("=" * 60)
    logger.info("Step 4: Creating / loading ACT policy")
    logger.info("=" * 60)

    if args.resume:
        resume_dir = Path(args.resume)
        if not resume_dir.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_dir}")
        logger.info(f"Resuming from checkpoint: {resume_dir}")
        policy = ACTPolicy.from_pretrained(str(resume_dir))
        preprocessor, postprocessor = make_pre_post_processors(
            cfg, pretrained_path=str(resume_dir), dataset_stats=ds_meta.stats,
        )
        logger.info("Policy and processors loaded from checkpoint")
    else:
        policy = ACTPolicy(cfg)
        preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=ds_meta.stats)
        logger.info("Policy and processors created")

    policy.train()
    policy.to(DEVICE)

    # 5. 构建 delta_timestamps（自动检测图像 key）
    logger.info("=" * 60)
    logger.info("Step 5: Building delta_timestamps")
    logger.info("=" * 60)
    fps = ds_meta.fps

    # 自动检测视觉特征 key（dtype="video"）
    image_keys = [
        k for k, ft in ds_meta.features.items()
        if ft.get("dtype") == "video"
    ]
    # 如果无视频特征，尝试检测 "dtype": "image"
    if not image_keys:
        image_keys = [
            k for k, ft in ds_meta.features.items()
            if ft.get("dtype") == "image"
        ]

    delta_timestamps = {
        # 所有图像的当前帧
        **{k: [0] for k in image_keys},
        # ACT 需要预测未来 chunk_size 个 action
        "action": [i / fps for i in range(actual_chunk_size)],
    }
    logger.info(f"Image keys: {image_keys}")
    logger.info(f"chunk_size={actual_chunk_size}, n_action_steps={actual_n_action_steps}")
    logger.info(f"Delta timestamps: action=[0..{actual_chunk_size - 1}]")

    # 6. 加载数据集
    logger.info("=" * 60)
    logger.info("Step 6: Loading dataset")
    logger.info("=" * 60)
    image_transforms = build_image_transforms()
    dataset = LeRobotDataset(
        DATASET_NAME,
        root=DATASET_ROOT,
        delta_timestamps=delta_timestamps,
        image_transforms=image_transforms,
    )
    logger.info(f"Dataset loaded: {dataset.num_frames} frames, {dataset.num_episodes} episodes")

    # 7. DataLoader
    logger.info("=" * 60)
    logger.info("Step 7: Creating DataLoader")
    logger.info("=" * 60)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=DEVICE.type != "cpu",
        drop_last=True,
    )
    logger.info(f"DataLoader: batch_size={BATCH_SIZE}, num_workers=4")

    # 8. 优化器
    logger.info("=" * 60)
    logger.info("Step 8: Setting up optimizer")
    logger.info("=" * 60)
    # 使用 ACT 内置的 optimizer param groups (backbone vs non-backbone)
    optim_params = policy.get_optim_params()
    optimizer = torch.optim.AdamW(optim_params)
    logger.info(f"Optimizer: AdamW with {len(optim_params)} param groups")

    # 9. 训练循环
    logger.info("=" * 60)
    logger.info("Step 9: Training loop")
    logger.info("=" * 60)

    step = 0
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        epoch_l1 = 0.0
        epoch_kld = 0.0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        for batch in pbar:
            # 移到 GPU
            batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Preprocess
            batch = preprocessor(batch)

            # Forward
            loss, loss_dict = policy.forward(batch)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            epoch_l1 += loss_dict.get("l1_loss", 0.0)
            epoch_kld += loss_dict.get("kld_loss", 0.0)
            n_batches += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "l1": f"{loss_dict.get('l1_loss', 0):.4f}",
                "kld": f"{loss_dict.get('kld_loss', 0):.4f}",
            })

            if step % LOG_FREQ == 0:
                logger.info(
                    f"step={step:05d} | loss={loss.item():.4f} | "
                    f"l1={loss_dict.get('l1_loss', 0):.4f} | kld={loss_dict.get('kld_loss', 0):.4f}"
                )

            if step > 0 and step % SAVE_FREQ == 0:
                ckpt_dir = OUTPUT_DIR / f"checkpoint_step_{step:06d}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                policy.save_pretrained(ckpt_dir)
                preprocessor.save_pretrained(ckpt_dir)
                postprocessor.save_pretrained(ckpt_dir)
                logger.info(f"Checkpoint saved to {ckpt_dir}")

            step += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_l1 = epoch_l1 / max(n_batches, 1)
        avg_kld = epoch_kld / max(n_batches, 1)
        logger.info(
            f"Epoch {epoch + 1} complete | avg_loss={avg_loss:.4f} | avg_l1={avg_l1:.4f} | avg_kld={avg_kld:.4f}"
        )

    # 10. 保存最终模型
    logger.info("=" * 60)
    logger.info("Step 10: Saving final model")
    logger.info("=" * 60)
    final_dir = OUTPUT_DIR / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(final_dir)
    preprocessor.save_pretrained(final_dir)
    postprocessor.save_pretrained(final_dir)
    logger.info(f"Final model saved to {final_dir}")

    logger.info("✅ Training complete!")


if __name__ == "__main__":
    main()
