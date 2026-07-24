#!/usr/bin/env python
"""Standalone cuRobo Piper IK diagnostic — runs inside Isaac Lab AppLauncher."""

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os, sys, tempfile, yaml
import torch
import numpy as np

from curobo.util_file import get_robot_configs_path, join_path, load_yaml, get_world_configs_path
from curobo.types.base import TensorDeviceType
from curobo.types.state import JointState
from curobo.types.math import Pose
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.geom.types import WorldConfig
from curobo.geom.sdf.world import CollisionCheckerType

# ── Setup ──────────────────────────────────────────────────────────────────
CUDA_ID = 0
tensor_args = TensorDeviceType(device=torch.device(f"cuda:{CUDA_ID}"), dtype=torch.float32)

# Resolve URDF
_piper_root = "/workspace/isaaclab/source/isaaclab_assets/isaaclab_assets/data/piper"
_urdf = os.path.join(_piper_root, "piper_description", "urdf",
                      "piper_description_v100_realsense_camera_v2.urdf")
os.environ.setdefault("ROS_PACKAGE_PATH", _piper_root)

# Load & patch piper.yml
data = load_yaml(join_path(get_robot_configs_path(), "piper.yml"))
data["robot_cfg"]["kinematics"]["urdf_path"] = _urdf
data["robot_cfg"]["kinematics"]["collision_spheres"] = "spheres/piper.yml"
kin = data["robot_cfg"]["kinematics"]
all_links = kin.get("link_names") or kin.get("mesh_link_names") or []
kin["collision_link_names"] = [n for n in all_links if n not in ("attached_object", "arm_base")]

tmp_dir = tempfile.mkdtemp(prefix="piper_ik_debug_")
yaml_path = os.path.join(tmp_dir, "piper.yml")
with open(yaml_path, "w") as f:
    yaml.safe_dump(data, f, sort_keys=False)

robot_cfg = data["robot_cfg"]
joint_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
retract_cfg = robot_cfg["kinematics"]["cspace"]["retract_config"]
print(f"Joint names : {joint_names}")
print(f"Retract cfg: {retract_cfg}")

# ── Build MotionGen ────────────────────────────────────────────────────────
world_cfg = WorldConfig.from_dict(
    load_yaml(join_path(get_world_configs_path(), "collision_table.yml")))
mg_cfg = MotionGenConfig.load_from_robot_config(
    robot_cfg, world_cfg,
    tensor_args=tensor_args,
    collision_checker_type=CollisionCheckerType.MESH,
    num_trajopt_seeds=48,
    num_graph_seeds=48,
    collision_activation_distance=0.0,
    position_threshold=0.005,
    rotation_threshold=0.05,
)
mg = MotionGen(mg_cfg)
print("MotionGen created OK")

mg.warmup(enable_graph=True, warmup_js_trajopt=False)
print("Warmup done")

plan_cfg = MotionGenPlanConfig(
    enable_graph=True, enable_graph_attempt=10,
    max_attempts=10, enable_finetune_trajopt=True,
)

# ── Test 1: identity plan from retract config ──────────────────────────────
retract_tensor = torch.tensor([retract_cfg], device=f"cuda:{CUDA_ID}", dtype=torch.float32)
js_retract = JointState(
    position=retract_tensor,
    joint_names=joint_names,
    tensor_args=tensor_args,
).get_ordered_joint_state(mg.kinematics.joint_names)

fk_retract = mg.rollout_fn.compute_kinematics(js_retract)
ee_retract = fk_retract.ee_pose
print(f"\n=== Test 1: Identity plan (retract config) ===")
print(f"EE pos : {ee_retract.position[0].tolist()}")
print(f"EE quat: {ee_retract.quaternion[0].tolist()}")

r1 = mg.plan_single(js_retract, ee_retract, plan_cfg)
print(f"Result : success={r1.success.item()}, status={r1.status}")

# ── Test 2: plan to slightly perturbed poses ────────────────────────────────
perturbations = [
    ("+X 2cm",   [0.02, 0.0, 0.0]),
    ("-X 2cm",   [-0.02, 0.0, 0.0]),
    ("+Y 2cm",   [0.0, 0.02, 0.0]),
    ("-Y 2cm",   [0.0, -0.02, 0.0]),
    ("+Z 2cm",   [0.0, 0.0, 0.02]),
    ("-Z 2cm",   [0.0, 0.0, -0.02]),   # retreat direction
    ("+X 5cm",   [0.05, 0.0, 0.0]),
    ("-Z 5cm",   [0.0, 0.0, -0.05]),
    ("+X 10cm",  [0.10, 0.0, 0.0]),
    ("diag 5cm", [0.03, 0.03, 0.03]),
]
print(f"\n=== Test 2: Small perturbations from retract ===")
for label, delta in perturbations:
    tgt = Pose(
        position=ee_retract.position.clone(),
        quaternion=ee_retract.quaternion.clone(),
    )
    tgt.position[0, 0] += delta[0]
    tgt.position[0, 1] += delta[1]
    tgt.position[0, 2] += delta[2]
    r = mg.plan_single(js_retract, tgt, plan_cfg)
    ok = r.success.item() if hasattr(r, "success") else "?"
    print(f"  {label:>10s}: success={ok}, status={r.status}")

# ── Test 3: try from random valid configs ───────────────────────────────────
print(f"\n=== Test 3: Identity from random configs (within limits) ===")
n_ok = 0
for i in range(10):
    # generate random config within [-1.5, 1.5] (roughly within joint limits)
    rand_pos = (torch.rand(8, device=f"cuda:{CUDA_ID}") * 2 - 1) * 1.5
    js_r = JointState(
        position=rand_pos.unsqueeze(0),
        joint_names=mg.kinematics.joint_names,
        tensor_args=tensor_args,
    )
    kin_r = mg.rollout_fn.compute_kinematics(js_r)
    ee_r = kin_r.ee_pose
    r = mg.plan_single(js_r, ee_r, plan_cfg)
    ok = r.success.item()
    if ok:
        n_ok += 1
    if i < 3 or ok:
        print(f"  Random {i}: pos={rand_pos[:3].tolist()}..., success={ok}, status={r.status}")
print(f"  Summary: {n_ok}/10 identity plans succeeded from random configs")

# ── Test 4: try with primitive collision checker ────────────────────────────
print(f"\n=== Test 4: PRIMITIVE collision checker ===")
mg_cfg2 = MotionGenConfig.load_from_robot_config(
    robot_cfg, world_cfg,
    tensor_args=tensor_args,
    collision_checker_type=CollisionCheckerType.PRIMITIVE,
    num_trajopt_seeds=48,
    num_graph_seeds=48,
    collision_activation_distance=0.0,
)
mg2 = MotionGen(mg_cfg2)
mg2.warmup(enable_graph=True, warmup_js_trajopt=False)
print("Primitive MotionGen created OK")

js2 = JointState(
    position=retract_tensor,
    joint_names=joint_names,
    tensor_args=tensor_args,
).get_ordered_joint_state(mg2.kinematics.joint_names)

fk2 = mg2.rollout_fn.compute_kinematics(js2)
ee2 = fk2.ee_pose

r_test = mg2.plan_single(js2, ee2, plan_cfg)
print(f"Identity (retract): success={r_test.success.item()}, status={r_test.status}")

for label, delta in perturbations[:6]:
    tgt = Pose(position=ee2.position.clone(), quaternion=ee2.quaternion.clone())
    tgt.position[0, 0] += delta[0]
    tgt.position[0, 1] += delta[1]
    tgt.position[0, 2] += delta[2]
    r = mg2.plan_single(js2, tgt, plan_cfg)
    ok = r.success.item() if hasattr(r, "success") else "?"
    print(f"  {label:>10s}: success={ok}, status={r.status}")

# ── Test 5: print joint limits ─────────────────────────────────────────────
print(f"\n=== Test 5: Joint limits ===")
jl = mg.kinematics.kinematics_config.joint_limits
if hasattr(jl, "position"):
    pos_lim = jl.position
    if isinstance(pos_lim, torch.Tensor):
        print(f"Position limits:\n{pos_lim}")
    else:
        print(f"Position limits: {pos_lim}")

# ── Cleanup ────────────────────────────────────────────────────────────────
import shutil
shutil.rmtree(tmp_dir, ignore_errors=True)
print("\nDone.")
simulation_app.close()
