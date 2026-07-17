#!/usr/bin/env python3
"""Quick test: compare model output with different image inputs."""
import torch, numpy as np, json, safetensors.torch as sft
from pathlib import Path
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

ckpt = Path('/workspace/isaaclab/logs/policy/D-SIM-PIPER-GRAB-0702-N50-K-V1-ACT/checkpoints/260000/pretrained_model')
dev = torch.device('cuda:0')

with open(ckpt / 'config.json') as f:
    rc = json.load(f)
rc.pop('input_features', None)
rc.pop('output_features', None)
for k in ['type','pretrained_path','pretrained_revision','push_to_hub','repo_id',
          'private','tags','license','device','use_amp','use_peft']:
    rc.pop(k, None)

cfg = ACTConfig(**rc)
cfg.input_features = {
    'observation.state': PolicyFeature(type=FeatureType('STATE'), shape=(7,)),
    'observation.images.front': PolicyFeature(type=FeatureType('VISUAL'), shape=(3, 720, 1280)),
    'observation.images.wrist': PolicyFeature(type=FeatureType('VISUAL'), shape=(3, 720, 1280)),
}
cfg.output_features = {'action': PolicyFeature(type=FeatureType('ACTION'), shape=(7,))}

sd = sft.load_file(str(ckpt / 'model.safetensors'))
policy = ACTPolicy(cfg)
policy.load_state_dict(sd)
policy.to(dev)
policy.eval()

pre, post = make_pre_post_processors(cfg, pretrained_path=str(ckpt))

s0 = torch.tensor([[0.0, 1.0, -0.6, 0.0, 1.22, 0.0, 0.05]], device=dev)

print(f'image_features: {policy.config.image_features}')
print(f'Preprocessor steps: {[type(s).__name__ for s in pre.steps]}')
print(f'Postprocessor steps: {[type(s).__name__ for s in post.steps]}')

# Test: exactly like eval — preprocess first, then build image_features list
z = torch.zeros(1, 3, 720, 1280, device=dev)
b1 = {'observation.state': s0, 'observation.images.front': z, 'observation.images.wrist': z}
print(f'Before pre: state shape={b1["observation.state"].shape}, front shape={b1["observation.images.front"].shape}')

b1 = pre(b1)
print(f'After pre keys: {list(b1.keys())}')
print(f'After pre state shape={b1.get("observation.state").shape if b1.get("observation.state") is not None else None}')
# Print normalized state
ns = b1.get('observation.state')
if ns is not None:
    print(f'Norm state: {np.array2string(ns.squeeze(0).cpu().numpy(), precision=4, suppress_small=True)}')

b1 = dict(b1)
if policy.config.image_features:
    b1['observation.images'] = [b1[k] for k in policy.config.image_features]
    print(f'image_features list: {[b1[k].shape for k in policy.config.image_features]}')

with torch.inference_mode():
    a1, _ = policy.model(b1)
a1 = post(a1[:, 0, :].cpu()).squeeze(0).numpy()
print(f'Action: {np.array2string(a1, precision=3, suppress_small=True)}')
print()
print(f'Eval step1:  [-0.04  0.95 -0.59 -0.    1.22 -0.02  0.05]')
print(f'Train step1: [ 0.    1.   -0.6   0.    1.22  0.    0.05]')
