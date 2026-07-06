"""Check what observation terms V1-A env actually has."""
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.managers import ObservationTermCfg

cfg = parse_env_cfg(task_name="Isaac-Piper-Grab-IK-Rel-Visuomotor-v1-A", device="cuda:0", num_envs=1)
obs_policy = cfg.observations.policy

print(f"Observations class: {type(obs_policy).__qualname__}")
print(f"Observations module: {type(obs_policy).__module__}")
print(f"concatenate_terms: {obs_policy.concatenate_terms}")
print()

print("Policy observation terms:")
for k, v in sorted(vars(obs_policy).items()):
    if isinstance(v, ObservationTermCfg):
        func_name = v.func.__name__ if hasattr(v.func, '__name__') else type(v.func).__name__
        print(f"  {k}: {func_name}")
