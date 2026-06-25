#!/usr/bin/env python3
"""
使用 LeRobot ACT 算法在 piper_grab_v1 数据集上训练策略。

数据: 100 episodes, 26376 frames, 30 FPS
  - observation.images.front: 1280x720 RGB
  - observation.images.wrist: 1280x720 RGB
  - observation.state: 63-dim (robot + object states)
  - action: 8-dim (IK delta pose + gripper)
  - task: "pick cube then grab bottle"

用法:
    conda activate lerobot
    python scripts/train_act_piper_grab.py                          # 从头训练
    python scripts/train_act_piper_grab.py --resume <checkpoint_dir>  # 续训
"""

import logging
import argparse
from pathlib import Path

import torch
import torchvision.transforms.v2 as T
from tqdm import tqdm

# --- 修复本地数据集无需联网验证 Hub 版本 ---
def _patch_get_safe_version():
    import lerobot.datasets.utils as ds_utils
    import lerobot.datasets.lerobot_dataset as lds

    def noop_get_safe_version(repo_id, version):
        return str(version)

    ds_utils.get_safe_version = noop_get_safe_version
    lds.get_safe_version = noop_get_safe_version


_patch_get_safe_version()

from lerobot.configs import FeatureType, NormalizationMode
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.act import ACTConfig, ACTPolicy
from lerobot.policies import make_pre_post_processors
from lerobot.utils.feature_utils import dataset_to_policy_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# 配置
# ============================================================
# ============================================================
# 配置 — 自动适配容器内/外路径
# ============================================================
def _find_dataset_root() -> Path:
    """自动查找数据集根目录，兼容容器内/外环境。"""
    candidates = [
        Path("/workspace/isaaclab/datasets/lerobot/piper_grab_v1"),  # Docker 容器内
        Path("/home/chenshengjia/Company/isaaclab/datasets/lerobot/piper_grab_v1"),  # 主机
    ]
    for p in candidates:
        if (p / "meta" / "info.json").exists():
            return p
    raise FileNotFoundError(f"Dataset not found. Tried: {candidates}")

# 命令行参数
parser = argparse.ArgumentParser(description="Train ACT policy on piper_grab_v1 dataset")
parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint directory to resume from")
args = parser.parse_args()

DATASET_ROOT = _find_dataset_root()
OUTPUT_DIR = DATASET_ROOT.parent / "piper_grab_v1_act_checkpoints"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# 训练参数
BATCH_SIZE = 8             # 双路 1280x720 图片，ResNet 占用大量显存
NUM_EPOCHS = 500           # 根据收敛情况调整
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
    ds_meta = LeRobotDatasetMetadata("piper_grab_v1", root=DATASET_ROOT)
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

    # 3. 创建 ACT 配置
    logger.info("=" * 60)
    logger.info("Step 3: Creating ACT config")
    logger.info("=" * 60)
    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        # ACT 核心参数
        chunk_size=CHUNK_SIZE,
        n_action_steps=N_ACTION_STEPS,
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
        pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
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
        logger.info("Policy loaded from checkpoint")

        # 加载 pre/post processors
        preprocessor = PolicyPreprocessor.from_pretrained(str(resume_dir))
        postprocessor = PolicyPostprocessor.from_pretrained(str(resume_dir))
        logger.info("Pre/post processors loaded from checkpoint")
    else:
        policy = ACTPolicy(cfg)
        # 创建 pre/post processors
        preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=ds_meta.stats)
        logger.info("Pre/post processors created")

    policy.train()
    policy.to(DEVICE)

    # 5. 构建 delta_timestamps
    logger.info("=" * 60)
    logger.info("Step 5: Building delta_timestamps")
    logger.info("=" * 60)
    fps = ds_meta.fps
    delta_timestamps = {
        # 当前时刻的图像
        "observation.images.front": [0],
        "observation.images.wrist": [0],
        # observation.state 不加 delta — ACT 内部自己处理 (n_obs_steps=1)
        # ACT 需要预测未来 chunk_size 个 action
        "action": [i / fps for i in range(CHUNK_SIZE)],
    }
    logger.info(f"Delta timestamps: {{'images': [0], 'action': [0..{CHUNK_SIZE - 1}]}}, state: raw")

    # 6. 加载数据集
    logger.info("=" * 60)
    logger.info("Step 6: Loading dataset")
    logger.info("=" * 60)
    image_transforms = build_image_transforms()
    dataset = LeRobotDataset(
        "piper_grab_v1",
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
