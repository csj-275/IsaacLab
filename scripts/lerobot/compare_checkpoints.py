import torch, numpy as np, json, safetensors.torch as sft, os
from pathlib import Path
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base = Path("/workspace/isaaclab/logs/policy/D-SIM-PIPER-GRAB-0702-N50-K-V1-ACT/checkpoints")

print(f"{'ckpt':<10} {'j1':>7} {'j2':>7} {'j3':>7} {'j4':>7} {'j5':>7} {'j6':>7} {'grip':>7}")
print(f"{'train':<10} {0.0:>7.3f} {1.0:>7.3f} {-0.6:>7.3f} {0.0:>7.3f} {1.22:>7.3f} {0.0:>7.3f} {0.05:>7.3f}")
print(f"{'eval':<10} {-0.04:>7.3f} {0.95:>7.3f} {-0.59:>7.3f} {0.0:>7.3f} {1.22:>7.3f} {-0.02:>7.3f} {0.05:>7.3f}")
print("-" * 70)

for ckpt_name in sorted(os.listdir(base)):
    ckpt_dir = base / ckpt_name / "pretrained_model"
    if not (ckpt_dir / "model.safetensors").exists():
        continue
    with open(ckpt_dir / "config.json") as f:
        rc = json.load(f)
    for k in ["input_features","output_features","type","pretrained_path","pretrained_revision",
              "push_to_hub","repo_id","private","tags","license","device","use_amp","use_peft"]:
        rc.pop(k, None)
    cfg = ACTConfig(**rc)
    cfg.input_features = {
        "observation.state": PolicyFeature(type=FeatureType("STATE"), shape=(7,)),
        "observation.images.front": PolicyFeature(type=FeatureType("VISUAL"), shape=(3, 720, 1280)),
        "observation.images.wrist": PolicyFeature(type=FeatureType("VISUAL"), shape=(3, 720, 1280)),
    }
    cfg.output_features = {"action": PolicyFeature(type=FeatureType("ACTION"), shape=(7,))}
    sd = sft.load_file(str(ckpt_dir / "model.safetensors"))
    policy = ACTPolicy(cfg)
    policy.load_state_dict(sd)
    policy.to(dev)
    policy.eval()
    pre, post = make_pre_post_processors(cfg, pretrained_path=str(ckpt_dir))
    s0 = torch.tensor([[0.0, 1.0, -0.6, 0.0, 1.22, 0.0, 0.05]])
    z = torch.zeros(1, 3, 720, 1280)
    b = {"observation.state": s0, "observation.images.front": z, "observation.images.wrist": z}
    b = pre(b)
    b = dict(b)
    b["observation.images"] = [b["observation.images.front"], b["observation.images.wrist"]]
    with torch.inference_mode():
        a, _ = policy.model(b)
    a = post(a[:, 0, :]).squeeze(0).numpy()
    print(f"{ckpt_name:<10} {a[0]:>7.3f} {a[1]:>7.3f} {a[2]:>7.3f} {a[3]:>7.3f} {a[4]:>7.3f} {a[5]:>7.3f} {a[6]:>7.3f}")
