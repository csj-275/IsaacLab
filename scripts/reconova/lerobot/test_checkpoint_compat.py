#!/usr/bin/env python3
"""快速验证 official lerobot checkpoint 是否兼容 eval_policy.py 的推理流程。

不需要 Isaac Sim，直接用假数据测试 forward pass。
"""
import json
import logging
from pathlib import Path

import torch
import torchvision.transforms.v2 as T

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
CHECKPOINT_DIR = "./datasets/policy/checkpoint"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

# ------------------------------------------------------------------
# 1. Load checkpoint config
# ------------------------------------------------------------------
ckpt_path = Path(CHECKPOINT_DIR)
with open(ckpt_path / "config.json") as f:
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

logger.info(f"Input  features: {list(cfg.input_features.keys())}")
logger.info(f"Output features: {list(cfg.output_features.keys())}")
logger.info(f"Image  features: {list(cfg.image_features.keys())}")
logger.info(f"Action shape: {cfg.output_features['action'].shape}")

# ------------------------------------------------------------------
# 2. Load policy + processors
# ------------------------------------------------------------------
import safetensors.torch as sft
state_dict = sft.load_file(str(ckpt_path / "model.safetensors"), device=str(DEVICE))
policy = ACTPolicy(cfg)
policy.load_state_dict(state_dict)
policy.to(DEVICE)
policy.eval()
logger.info(f"Policy loaded on {DEVICE}, params={sum(p.numel() for p in policy.parameters()):,}")

preprocessor, postprocessor = make_pre_post_processors(cfg, pretrained_path=str(ckpt_path))
logger.info("Preprocessor & Postprocessor loaded from checkpoint")

# ------------------------------------------------------------------
# 3. Auto-detect expected dims from preprocessor stats
# ------------------------------------------------------------------
import safetensors
preproc_path = Path(CHECKPOINT_DIR)
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

logger.info(f"Expected state dim: {expected_state_dim}")
logger.info(f"Expected action dim: {expected_action_dim}")
assert expected_state_dim == 32, f"Unexpected state dim: {expected_state_dim}"

# ------------------------------------------------------------------
# 4. Build fake batch (matching eval_policy.py's build functions)
# ------------------------------------------------------------------
IMG_RESIZE = (224, 224)
transforms = T.Compose([
    T.Resize(IMG_RESIZE, antialias=True),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Simulate env output: images as uint8 (H,W,3)
fake_img = torch.randint(0, 255, (720, 1280, 3), dtype=torch.uint8)

# Build image batch (simulating build_image_batch)
batch = {}
for feat_key in ["observation.images.front", "observation.images.wrist"]:
    img = fake_img.float().unsqueeze(0)  # (1, H, W, 3)
    img = img.permute(0, 3, 1, 2)        # (1, 3, H, W)
    img = transforms(img.to(DEVICE))
    batch[feat_key] = img
    logger.info(f"  {feat_key}: shape={batch[feat_key].shape}")

# Simulate state: zeros (32-dim)
STATE_KEY = "observation.state"
batch[STATE_KEY] = torch.zeros(1, expected_state_dim, device=DEVICE)
logger.info(f"  {STATE_KEY}: shape={batch[STATE_KEY].shape}")

# ------------------------------------------------------------------
# 5. Preprocess → Forward → Postprocess
# ------------------------------------------------------------------
# Preprocess
batch = preprocessor(batch)
logger.info(f"After preprocessor: batch keys = {sorted(batch.keys())}")

# Re-key images (matching eval_policy.py line 307-309)
if policy.config.image_features:
    batch = dict(batch)
    batch["observation.images"] = [batch[k] for k in policy.config.image_features]
    logger.info(f"After re-key: observation.images = [{batch['observation.images'][0].shape}, {batch['observation.images'][1].shape}]")

# Forward
with torch.inference_mode():
    actions_hat, _ = policy.model(batch)
logger.info(f"Model output (actions_hat): shape={actions_hat.shape}")

# Postprocess
action = actions_hat[:, 0, :].cpu()
logger.info(f"Before postprocess: action shape={action.shape}, min={action.min():.4f}, max={action.max():.4f}")
action = postprocessor(action).squeeze(0).numpy()
logger.info(f"After  postprocess: action shape={action.shape}, values=[{', '.join(f'{v:.4f}' for v in action)}]")

# ------------------------------------------------------------------
# 6. Sanity checks
# ------------------------------------------------------------------
assert action.shape == (expected_action_dim,), f"Wrong action shape: {action.shape}, expected ({expected_action_dim},)"
assert not (action == 0).all(), "WARNING: All-zero action! (could be expected with zero-state input)"
logger.info(f"Action range: min={action.min():.4f}, max={action.max():.4f}")

# Check if action is in reasonable physical range (abs values should not be huge)
if abs(action).max() > 100:
    logger.error(f"❌ Action values are huge ({abs(action).max():.2f})! Denormalization might be broken.")
else:
    logger.info(f"✅ Action values in reasonable range (max abs = {abs(action).max():.4f})")

logger.info("\n✅ All checks passed! Policy is compatible with eval_policy.py inference pipeline.")
logger.info("Ready to run full evaluation in Isaac Sim container.")
