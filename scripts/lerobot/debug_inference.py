#!/usr/bin/env python3
"""Debug: run model inference on a training data sample and compare output vs ground truth.

Usage:
    ./isaaclab.sh -p scripts/lerobot/debug_inference.py \\
        --checkpoint-dir ./logs/policy/D-SIM-PIPER-GRAB-0702-N100-K-V1-ACT/checkpoints/last/pretrained_model/ \\
        --dataset-dir ./datasets/lerobot/D-SIM-PIPER-GRAB-0702-N100-K-V1 \\
        --num-samples 5
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint-dir", type=str, required=True)
parser.add_argument("--dataset-dir", type=str, required=True)
parser.add_argument("--num-samples", type=int, default=5)
parser.add_argument("--device", type=str, default="cuda")
args = parser.parse_args()

device = torch.device(args.device)

# ---------------------------------------------------------------------------
# Load policy
# ---------------------------------------------------------------------------
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

ckpt_path = Path(args.checkpoint_dir)
with open(ckpt_path / "config.json") as f:
    raw_config = json.load(f)

input_features_raw = raw_config.pop("input_features", {})
output_features_raw = raw_config.pop("output_features", {})

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

import safetensors.torch as sft
state_dict = sft.load_file(str(ckpt_path / "model.safetensors"), device=str(device))
policy = ACTPolicy(cfg)
policy.load_state_dict(state_dict)
policy.to(device)
policy.eval()

preprocessor, postprocessor = make_pre_post_processors(cfg, pretrained_path=str(ckpt_path))
print(f"Model loaded. Input: {list(cfg.input_features.keys())}, Output: {list(cfg.output_features.keys())}")

# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------
import glob
dataset_dir = args.dataset_dir
with open(os.path.join(dataset_dir, "meta", "info.json")) as f:
    info = json.load(f)

meta_files = sorted(glob.glob(os.path.join(dataset_dir, "meta", "episodes", "chunk-*", "file-*.parquet")))
episodes_meta = pd.concat([pd.read_parquet(f) for f in meta_files], ignore_index=True)

data_files = sorted(glob.glob(os.path.join(dataset_dir, "data", "chunk-*", "file-*.parquet")))
all_data = pd.concat([pd.read_parquet(f) for f in data_files], ignore_index=True)

# ---------------------------------------------------------------------------
# Sample and test
# ---------------------------------------------------------------------------
import random
random.seed(42)

# Pick random episodes, then pick random frames within them
ep_indices = sorted(random.sample(range(len(episodes_meta)), min(args.num_samples, len(episodes_meta))))

print(f"\n{'='*60}")
print(f"Testing {len(ep_indices)} samples from dataset")

STATE_KEY = "observation.state"
IMG_KEYS_MAP = {
    "observation.images.front": "table_cam",
    "observation.images.wrist": "wrist_cam",
}

video_dir = Path(dataset_dir) / "videos"

for i, ep_idx in enumerate(ep_indices):
    ep_meta = episodes_meta.iloc[ep_idx]
    ep_num = int(ep_meta["episode_index"])

    # Get a random frame from this episode (not frame 0, which is identity)
    ep_data = all_data[all_data["episode_index"] == ep_num].sort_values("frame_index")
    if len(ep_data) < 10:
        continue
    frame_idx = random.randint(5, len(ep_data) - 1)
    row = ep_data.iloc[frame_idx]

    # Build observation.state from dataset
    raw_state = np.array(row[STATE_KEY], dtype=np.float32)
    state_tensor = torch.from_numpy(raw_state).unsqueeze(0).to(device)

    # Build ground truth action
    gt_action = np.array(row["action"], dtype=np.float32)

    # Build image batch — load from parquet if stored inline, or from video files
    batch = {STATE_KEY: state_tensor}

    for img_key, cam_key in IMG_KEYS_MAP.items():
        if img_key in cfg.input_features:
            # Try to find image in observation columns or video files
            cam_val = row.get(img_key, None)
            if cam_val is not None and isinstance(cam_val, np.ndarray) and cam_val.size > 0:
                img = cam_val
            else:
                # Fallback: load from video (LeRobot v3 format: file-{ep:03d}.mp4)
                import cv2
                video_path = video_dir / img_key / "chunk-000" / f"file-{ep_num:03d}.mp4"
                if video_path.exists():
                    cap = cv2.VideoCapture(str(video_path))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame_bgr = cap.read()
                    cap.release()
                    if ret:
                        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    else:
                        print(f"  [WARN] Failed to read frame from {video_path}, using zeros")
                        img = np.zeros((720, 1280, 3), dtype=np.uint8)
                else:
                    print(f"  [WARN] No image for {img_key} ep={ep_num} frame={frame_idx}, using zeros")
                    img = np.zeros((720, 1280, 3), dtype=np.uint8)

            img_tensor = torch.from_numpy(img).float() / 255.0  # uint8 → float32 [0,1]
            if img_tensor.ndim == 3:
                img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
            batch[img_key] = img_tensor.to(device)

    # Preprocess
    batch = preprocessor(batch)
    if policy.config.image_features:
        batch = dict(batch)
        batch["observation.images"] = [batch[k] for k in policy.config.image_features]

    # Infer
    with torch.inference_mode():
        actions_hat, _ = policy.model(batch)

    pred_action = actions_hat[:, 0, :].cpu()
    pred_action = postprocessor(pred_action).squeeze(0).numpy()

    # Compare
    abs_err = np.abs(pred_action - gt_action)
    rel_err = abs_err / (np.abs(gt_action) + 1e-8)

    joint_names = ["j1", "j2", "j3", "j4", "j5", "j6", "grip"]
    print(f"\n--- Sample {i+1}: ep={ep_num}, frame={frame_idx} ---")
    print(f"  state:  {np.array2string(raw_state, precision=3, suppress_small=True, max_line_width=80)}")
    print(f"  gt:     {np.array2string(gt_action, precision=3, suppress_small=True, max_line_width=80)}")
    print(f"  pred:   {np.array2string(pred_action, precision=3, suppress_small=True, max_line_width=80)}")
    print(f"  |err|:  {np.array2string(abs_err, precision=3, suppress_small=True, max_line_width=80)}")
    print(f"  |err|%: {np.array2string(rel_err * 100, precision=1, suppress_small=True, max_line_width=80)}")
    for j, name in enumerate(joint_names):
        pred_val = pred_action[j]
        gt_val = gt_action[j]
        state_val = raw_state[j]
        print(f"  {name}: state={state_val:.4f}, gt={gt_val:.4f}, pred={pred_val:.4f}, "
              f"diff_gt={pred_val - gt_val:+.4f}, diff_state={pred_val - state_val:+.4f}")

print(f"\n{'='*60}")
print("Done.")
