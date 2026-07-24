#!/usr/bin/env python
"""Franka IK diagnostic — comprehensive test."""

from isaaclab.app import AppLauncher
import argparse
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os, tempfile, yaml, torch, sys
from curobo.util_file import get_robot_configs_path, join_path, load_yaml, get_world_configs_path
from curobo.types.base import TensorDeviceType
from curobo.types.state import JointState
from curobo.types.math import Pose
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.geom.types import WorldConfig
from curobo.geom.sdf.world import CollisionCheckerType
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, retrieve_file_path

CUDA_ID = 0
tensor_args = TensorDeviceType(device=torch.device(f"cuda:{CUDA_ID}"), dtype=torch.float32)

# ── Download Franka URDF & build config ───────────────────────────────────
urdf_url = f"{ISAACLAB_NUCLEUS_DIR}/Controllers/SkillGenAssets/FrankaPanda/franka_panda.urdf"
furdf = retrieve_file_path(urdf_url, force_download=True)
sys.stdout.write(f"Franka URDF: {furdf}\n")

data = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
data["robot_cfg"]["kinematics"]["urdf_path"] = furdf
data["robot_cfg"]["kinematics"]["collision_spheres"] = "spheres/franka_mesh.yml"

# Strip locked joints from cspace (gripper fingers)
kin = data["robot_cfg"]["kinematics"]
lock_joints = kin.get("lock_joints", {})
cspace = kin.get("cspace", {})
if lock_joints and cspace:
    cjn = cspace["joint_names"]
    keep_idx = [i for i, n in enumerate(cjn) if n not in lock_joints]
    for key in ("joint_names", "retract_config", "null_space_weight", "cspace_distance_weight"):
        if key in cspace:
            cspace[key] = [cspace[key][i] for i in keep_idx]
    sys.stdout.write(f"cspace joints (after strip): {cspace['joint_names']} ({len(cspace['joint_names'])})\n")
    sys.stdout.write(f"retract_config: {cspace['retract_config']}\n")

# Auto-populate collision_link_names
all_links = kin.get("link_names") or kin.get("mesh_link_names") or []
kin["collision_link_names"] = [n for n in all_links if n != "attached_object"]

tmp_dir = tempfile.mkdtemp(prefix="franka_test_")
with open(os.path.join(tmp_dir, "franka.yml"), "w") as f:
    yaml.safe_dump(data, f, sort_keys=False)

world_cfg = WorldConfig.from_dict(load_yaml(join_path(get_world_configs_path(), "collision_table.yml")))

# ── Build MotionGen ───────────────────────────────────────────────────────
sys.stdout.write("\n=== Building MotionGen (MESH) ===\n")
mg_cfg = MotionGenConfig.load_from_robot_config(
    data["robot_cfg"], world_cfg, tensor_args=tensor_args,
    collision_checker_type=CollisionCheckerType.MESH,
    num_trajopt_seeds=24, num_graph_seeds=24,
    collision_activation_distance=0.0,
)
mg = MotionGen(mg_cfg)
mg.warmup(enable_graph=True, warmup_js_trajopt=False)
sys.stdout.write(f"Kin joints: {mg.kinematics.joint_names} ({len(mg.kinematics.joint_names)} DOF)\n")
sys.stdout.write(f"EE link: {mg.kinematics.kinematics_config.ee_link}\n")

# ── FK at retract config ──────────────────────────────────────────────────
rc = cspace["retract_config"]
jn = cspace["joint_names"]
rt = torch.tensor([rc], device=f"cuda:{CUDA_ID}", dtype=torch.float32)
js = JointState(position=rt, joint_names=jn, tensor_args=tensor_args)
js = js.get_ordered_joint_state(mg.kinematics.joint_names)
fk = mg.rollout_fn.compute_kinematics(js)
ee = fk.ee_pose
sys.stdout.write(f"FK EE pos:  {ee.position[0].tolist()}\n")
sys.stdout.write(f"FK EE quat: {ee.quaternion[0].tolist()}\n")

# ── Test 1: IK identity (retract → retract) ───────────────────────────────
sys.stdout.write("\n=== Test 1: IK solve_single identity ===\n")
ik = mg.ik_solver
for n_seeds in [1, 5, 10, 50]:
    r = ik.solve_single(ee, js.position, num_seeds=n_seeds)
    s = r.success.item()
    err = f"pos_err={r.position_error.item():.6f}, rot_err={r.rotation_error.item():.6f}" if s else ""
    sys.stdout.write(f"  n={n_seeds:>3}: success={s} {err}\n")
    sys.stdout.flush()

# ── Test 2: IK with perturbations ─────────────────────────────────────────
sys.stdout.write("\n=== Test 2: IK perturbations ===\n")
for label, delta in [
    ("+X2cm", [0.02, 0, 0]), ("-X2cm", [-0.02, 0, 0]),
    ("+Y2cm", [0, 0.02, 0]), ("-Y2cm", [0, -0.02, 0]),
    ("+Z2cm", [0, 0, 0.02]), ("-Z2cm", [0, 0, -0.02]),
    ("+X5cm", [0.05, 0, 0]), ("-Z5cm", [0, 0, -0.05]),
    ("+X10cm", [0.10, 0, 0]), ("diag5cm", [0.03, 0.03, 0.03]),
]:
    tgt = Pose(position=ee.position.clone(), quaternion=ee.quaternion.clone())
    tgt.position[0, 0] += delta[0]
    tgt.position[0, 1] += delta[1]
    tgt.position[0, 2] += delta[2]
    r = ik.solve_single(tgt, js.position, num_seeds=50)
    s = r.success.item()
    err = f"pos_err={r.position_error.item():.6f}" if s else ""
    sys.stdout.write(f"  {label:>10s}: success={s} {err}\n")
    sys.stdout.flush()

# ── Test 3: plan_single (full motion planning) ─────────────────────────────
sys.stdout.write("\n=== Test 3: plan_single ===\n")
pc = MotionGenPlanConfig(enable_graph=True, enable_graph_attempt=10,
                         max_attempts=10, enable_finetune_trajopt=True)
r = mg.plan_single(js, ee, pc)
sys.stdout.write(f"  Identity plan: success={r.success.item()}, status={r.status}\n")

for label, delta in [
    ("+X2cm", [0.02, 0, 0]), ("-X2cm", [-0.02, 0, 0]),
    ("+X5cm", [0.05, 0, 0]), ("-Z5cm", [0, 0, -0.05]),
    ("+X10cm", [0.10, 0, 0]),
]:
    tgt = Pose(position=ee.position.clone(), quaternion=ee.quaternion.clone())
    tgt.position[0, 0] += delta[0]
    tgt.position[0, 1] += delta[1]
    tgt.position[0, 2] += delta[2]
    try:
        r = mg.plan_single(js, tgt, pc)
        sys.stdout.write(f"  {label:>10s}: success={r.success.item()}, status={r.status}\n")
    except Exception as e:
        sys.stdout.write(f"  {label:>10s}: ERROR: {type(e).__name__}: {e}\n")
    sys.stdout.flush()

# ── Test 4: Random config IK ──────────────────────────────────────────────
sys.stdout.write("\n=== Test 4: IK from random configs ===\n")
ok = 0
for i in range(20):
    rand_j = (torch.rand(7, device=f"cuda:{CUDA_ID}") * 2 - 1) * 2.0
    js_r = JointState(position=rand_j.unsqueeze(0), joint_names=mg.kinematics.joint_names,
                      tensor_args=tensor_args)
    fk_r = mg.rollout_fn.compute_kinematics(js_r)
    ee_r = fk_r.ee_pose
    r = ik.solve_single(ee_r, js_r.position, num_seeds=20)
    s = r.success.item()
    if s: ok += 1
sys.stdout.write(f"  Summary: {ok}/20 identity IK succeeded from random configs\n")

# ── Cleanup ───────────────────────────────────────────────────────────────
import shutil
shutil.rmtree(tmp_dir, ignore_errors=True)
sys.stdout.write("\nDone.\n")
simulation_app.close()
