#!/usr/bin/env python
"""cuRobo Piper IK diagnostic v2 — test pure IK (bypass plan_single)."""

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

CUDA_ID = 0
tensor_args = TensorDeviceType(device=torch.device(f"cuda:{CUDA_ID}"), dtype=torch.float32)

_piper_root = "/workspace/isaaclab/source/isaaclab_assets/isaaclab_assets/data/piper"
_urdf = os.path.join(_piper_root, "piper_description", "urdf",
                      "piper_description_v100_realsense_camera_v2.urdf")
os.environ.setdefault("ROS_PACKAGE_PATH", _piper_root)

data = load_yaml(join_path(get_robot_configs_path(), "piper.yml"))
data["robot_cfg"]["kinematics"]["urdf_path"] = _urdf
data["robot_cfg"]["kinematics"]["collision_spheres"] = "spheres/piper.yml"
kin = data["robot_cfg"]["kinematics"]
all_links = kin.get("link_names") or kin.get("mesh_link_names") or []
kin["collision_link_names"] = [n for n in all_links if n not in ("attached_object", "arm_base")]

tmp_dir = tempfile.mkdtemp(prefix="piper_ik2_")
yaml_path = os.path.join(tmp_dir, "piper.yml")
with open(yaml_path, "w") as f:
    yaml.safe_dump(data, f, sort_keys=False)

robot_cfg = data["robot_cfg"]
joint_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
retract_cfg = robot_cfg["kinematics"]["cspace"]["retract_config"]

# ── Use PRIMITIVE to bypass warp.torch issues ──────────────────────────────
world_cfg = WorldConfig.from_dict(
    load_yaml(join_path(get_world_configs_path(), "collision_table.yml")))

mg_cfg = MotionGenConfig.load_from_robot_config(
    robot_cfg, world_cfg,
    tensor_args=tensor_args,
    collision_checker_type=CollisionCheckerType.PRIMITIVE,
    num_trajopt_seeds=48,
    num_graph_seeds=48,
    collision_activation_distance=0.0,
)
mg = MotionGen(mg_cfg)
mg.warmup(enable_graph=True, warmup_js_trajopt=False)
print("MotionGen (PRIMITIVE) created + warmed up")

retract_t = torch.tensor([retract_cfg], device=f"cuda:{CUDA_ID}", dtype=torch.float32)
js = JointState(
    position=retract_t, joint_names=joint_names, tensor_args=tensor_args,
).get_ordered_joint_state(mg.kinematics.joint_names)

# ── Test A: Direct IK solve ────────────────────────────────────────────────
print("\n=== Test A: Direct IK (inverse kinematics) ===")
fk = mg.rollout_fn.compute_kinematics(js)
ee = fk.ee_pose
print(f"FK EE pos:  {ee.position[0].tolist()}")
print(f"FK EE quat: {ee.quaternion[0].tolist()}")

# Try cuRobo's IK solver directly
try:
    ik_result = mg.ik_solver.solve(
        goal_pose=ee,
        retract_config=js.position,
        num_seeds=100,
    )
    print(f"Direct IK: success={ik_result.success.item()}, num_results={ik_result.num_results}")
    if ik_result.success.item():
        print(f"  IK joint pos: {ik_result.solution.position[0].tolist()}")
        print(f"  FK of IK result: ...")
except Exception as e:
    print(f"Direct IK error: {e}")

# ── Test B: Try with disable_graph ──────────────────────────────────────────
print("\n=== Test B: plan_single with graph DISABLED ===")
pc_no_graph = MotionGenPlanConfig(
    enable_graph=False,
    enable_graph_attempt=0,
    max_attempts=20,
    enable_finetune_trajopt=True,
)
try:
    r = mg.plan_single(js, ee, pc_no_graph)
    print(f"No-graph identity: success={r.success.item()}, status={r.status}")
except Exception as e:
    print(f"No-graph plan_single error: {e}")

# ── Test C: plan_single_js (joint-space planning) ──────────────────────────
print("\n=== Test C: plan_single_js (joint-space) ===")
rand_goal = retract_t.clone()
rand_goal[0, 0] += 0.1  # slightly different joint config
js_goal = JointState(
    position=rand_goal, joint_names=joint_names, tensor_args=tensor_args,
).get_ordered_joint_state(mg.kinematics.joint_names)
try:
    r = mg.plan_single_js(js, js_goal, pc_no_graph)
    print(f"JS plan: success={r.success.item()}, status={r.status}")
except Exception as e:
    print(f"plan_single_js error: {e}")

# ── Test D: Check if searching kinematics works ─────────────────────────────
print("\n=== Test D: Graph search debug ===")
print(f"Kinematics joint names: {mg.kinematics.joint_names}")
print(f"Kinematics ee_link: {mg.kinematics.kinematics_config.ee_link}")
print(f"Num seeds: graphtrajopt={mg_cfg.num_graph_seeds}, trajopt={mg_cfg.num_trajopt_seeds}")

# Check what the graph planner finds
for n_seeds in [10, 50, 200, 500]:
    try:
        r = mg.plan_single(js, ee, MotionGenPlanConfig(
            enable_graph=True, enable_graph_attempt=5,
            max_attempts=5, enable_finetune_trajopt=False,
        ))
        print(f"  Graph-only (n_seeds={n_seeds}): success={r.success.item()}, status={r.status}")
    except Exception as e:
        print(f"  Graph-only ({n_seeds}): error: {e}")

# ── Test E: Joint limit verification ────────────────────────────────────────
print("\n=== Test E: Joint limits ===")
try:
    jl = mg.kinematics.kinematics_config.joint_limits
    if hasattr(jl, "position"):
        pl = jl.position
        print(f"Position limits shape: {pl.shape if hasattr(pl, 'shape') else type(pl)}")
        print(f"Position limits: {pl}")
except Exception as e:
    print(f"Joint limit error: {e}")

print(f"\nRetract at limits? retract={retract_cfg}")

import shutil
shutil.rmtree(tmp_dir, ignore_errors=True)
print("Done.")
simulation_app.close()
