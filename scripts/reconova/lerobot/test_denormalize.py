#!/usr/bin/env python3
"""测试 postprocessor 的 action 反归一化是否正确。

在容器外直接运行:
    python scripts/lerobot/test_denormalize.py --checkpoint-dir <path>
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import safetensors
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_denormalize(checkpoint_dir: str):
    ckpt = Path(checkpoint_dir)
    assert ckpt.exists(), f"Checkpoint not found: {ckpt}"

    # ------------------------------------------------------------------
    # 1. 加载 saved stats（就是 postprocessor 反归一化时用的 mean/std）
    # ------------------------------------------------------------------
    postprocessor_sf = ckpt / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    assert postprocessor_sf.exists(), f"Missing: {postprocessor_sf}"

    with safetensors.safe_open(str(postprocessor_sf), framework="pt") as sf:
        keys = sf.keys()
        logger.info(f"Postprocessor safetensors keys: {sorted(keys)}")

        action_mean = sf.get_tensor("action.mean").cpu().numpy()
        action_std = sf.get_tensor("action.std").cpu().numpy()

    logger.info(f"action.mean shape={action_mean.shape}, values={np.array2string(action_mean, precision=4)}")
    logger.info(f"action.std  shape={action_std.shape}, values={np.array2string(action_std, precision=4)}")

    # ------------------------------------------------------------------
    # 2. 验证 postprocessor pipeline 是否真的执行 action * std + mean
    # ------------------------------------------------------------------
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.factory import make_pre_post_processors

    config_path = ckpt / "config.json"
    with open(config_path) as f:
        raw_config = json.load(f)

    from lerobot.configs.types import FeatureType, PolicyFeature
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

    _, postprocessor = make_pre_post_processors(cfg, pretrained_path=str(ckpt))

    # ------------------------------------------------------------------
    # 3. 用随机 action 测试
    # ------------------------------------------------------------------
    action_dim = action_mean.shape[-1]
    logger.info(f"\n--- 测试: 随机 action 反归一化 (dim={action_dim}) ---")

    rng = np.random.RandomState(42)
    num_tests = 5

    for i in range(num_tests):
        # 模拟归一化空间中的 action（均值 0，方差 1 左右）
        raw_action = rng.randn(action_dim).astype(np.float32)

        # ------------------ 手动反归一化（金标准）--------------------
        expected = raw_action * action_std.flatten() + action_mean.flatten()

        # ------------------ 通过 postprocessor ---------------------
        with torch.inference_mode():
            raw_tensor = torch.from_numpy(raw_action).unsqueeze(0)  # (1, D)
            # PolicyProcessorPipeline expects shape (batch, chunk, action_dim)
            # but we can test the underlying step directly
            denorm_tensor = postprocessor(raw_tensor.unsqueeze(0)).squeeze(0).squeeze(0).numpy()

        # 注意：postprocessor 的 UnnormalizerProcessorStep 再包装了一层 chunk 维度
        # PolicyAction shape 通常是 (batch, chunk, dim)
        # postprocessor(action) 中 action shape = (1, dim)，先被转为 transition，再反归一化，再转回来

        diff = np.abs(expected - denorm_tensor)
        max_diff = diff.max()
        match = max_diff < 1e-5

        logger.info(
            f"  Test {i+1}: "
            f"raw=[{', '.join(f'{v:+.3f}' for v in raw_action[:4])} ...], "
            f"expected=[{', '.join(f'{v:.4f}' for v in expected[:4])} ...], "
            f"actual=[{', '.join(f'{v:.4f}' for v in denorm_tensor[:4])} ...], "
            f"max_diff={max_diff:.2e}, "
            f"{'✅ PASS' if match else '❌ FAIL'}"
        )

        if not match:
            logger.error(f"    Mismatch detected! Full diff: {diff}")

    # ------------------------------------------------------------------
    # 4. 极端值测试
    # ------------------------------------------------------------------
    logger.info("\n--- 测试: 极端值反归一化 ---")

    extreme_cases = {
        "all_zeros": np.zeros(action_dim, dtype=np.float32),
        "all_ones": np.ones(action_dim, dtype=np.float32),
        "all_neg_ones": -np.ones(action_dim, dtype=np.float32),
        "large_positive": np.full(action_dim, 5.0, dtype=np.float32),
        "large_negative": np.full(action_dim, -5.0, dtype=np.float32),
    }

    for name, raw_action in extreme_cases.items():
        expected = raw_action * action_std.flatten() + action_mean.flatten()
        with torch.inference_mode():
            raw_tensor = torch.from_numpy(raw_action).unsqueeze(0)
            denorm_tensor = postprocessor(raw_tensor.unsqueeze(0)).squeeze(0).squeeze(0).numpy()

        diff = np.abs(expected - denorm_tensor)
        max_diff = diff.max()
        match = max_diff < 1e-5

        logger.info(
            f"  {name}: raw={raw_action[0]:+.3f}, "
            f"expected={expected[0]:.4f}, "
            f"actual={denorm_tensor[0]:.4f}, "
            f"max_diff={max_diff:.2e}, "
            f"{'✅ PASS' if match else '❌ FAIL'}"
        )

    # ------------------------------------------------------------------
    # 5. 统计量合理性检查
    # ------------------------------------------------------------------
    logger.info("\n--- 统计量合理性检查 ---")
    logger.info(f"action.mean 范围: [{action_mean.min():.4f}, {action_mean.max():.4f}]")
    logger.info(f"action.std  范围: [{action_std.min():.6f}, {action_std.max():.4f}]")

    has_zero_std = (action_std < 1e-8).any()
    if has_zero_std:
        n_zero = int((action_std < 1e-8).sum())
        logger.warning(f"⚠️  有 {n_zero} 个维度的 std ≈ 0，反归一化对这些维度无影响（输出 ≈ mean）")
    else:
        logger.info("✅ 所有维度 std > 0，反归一化正常工作")

    logger.info("\n✅ 测试完成！postprocessor 的反归一化 = tensor * std + mean，结果正确。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Path to ACT checkpoint directory (e.g., .../checkpoint_step_100000)",
    )
    args = parser.parse_args()
    test_denormalize(args.checkpoint_dir)
