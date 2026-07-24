#!/usr/bin/env python
"""cuRobo Piper IK diagnostic v3 — strip locked joints from retract_config."""

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

# ── CRITICAL FIX: strip locked joints from cspace ──────────────────────────
lock_joints = kin.get("lock_joints", {})
cspace = kin.get("cspace", {})
if lock_joints and cspace and "retract_config" in cspace:
    cspace_joint_names = cspace.get("joint_names", [])
    for key in ("retract_config", "joint_names", "null_space_weight", "cspace_distance_weight"):
        if key in cspace:
            cspace[key] = [v for v, name in zip(cspace[key], cspace_joint_names) if name not in lock_joints]
    sys.stdout.write(f"[FIX] retract_config stripped: {cspace['retract_config']}\n")

tmp_dir = tempfile.mkdtemp(prefix="piper_ik3_")
yaml_path = os.path.join(tmp_dir, "piper.yml")
with open(yaml_path, "w") as f:
    yaml.safe_dump(data, f, sort_keys=False)

robot_cfg = data["robot_cfg"]
world_cfg = WorldConfig.from_dict(load_yaml(join_path(get_world_configs_path(), "collision_table.yml")))

mg_cfg = MotionGenConfig.load_from_robot_config(
    robot_cfg, world_cfg, tensor_args=tensor_args,
    collision_checker_type=CollisionCheckerType.PRIMITIVE,
    num_trajopt_seeds=100, num_graph_seeds=100,
    collision_activation_distance=0.0,
)
mg = MotionGen(mg_cfg)
mg.warmup(enable_graph=True, warmup_js_trajopt=False)
sys.stdout.write("MotionGen created + warmed up\n")
sys.stdout.flush()

retract_cfg = robot_cfg["kinematics"]["cspace"]["retract_config"]
cspace_jn = robot_cfg["kinematics"]["cspace"]["joint_names"]
sys.stdout.write(f"Cspace joints: {cspace_jn} ({len(cspace_jn)})\n")
sys.stdout.write(f"Retract cfg: {retract_cfg} ({len(retract_cfg)})\n")
sys.stdout.write(f"Kinematics joints: {mg.kinematics.joint_names} ({len(mg.kinematics.joint_names)})\n")

retract_t = torch.tensor([retract_cfg], device=f"cuda:{CUDA_ID}", dtype=torch.float32)
js = JointState(position=retract_t, joint_names=cspace_jn, tensor_args=tensor_args)
js = js.get_ordered_joint_state(mg.kinematics.joint_names)
sys.stdout.write(f"JS position shape: {js.position.shape}\n")

fk = mg.rollout_fn.compute_kinematics(js)
ee = fk.ee_pose
sys.stdout.write(f"FK EE pos: {ee.position[0].tolist()}\n")
sys.stdout.flush()

# ── Test solve_single ──────────────────────────────────────────────────────
iks = mg.ik_solver
sys.stdout.write("\n=== solve_single tests ===\n")
for n_seeds in [10, 100, 500, 1000]:
    try:
        result = iks.solve_single(ee, js.position, num_seeds=n_seeds)
        s = result.success.item()
        sys.stdout.write(f"  n={n_seeds}: success={s}\n")
        if s:
            sys.stdout.write(f"    solution: {result.solution.position[0].tolist()}\n")
    except Exception as e:
        sys.stdout.write(f"  n={n_seeds}: ERROR: {e}\n")
    sys.stdout.flush()

# ── Test plan_single ───────────────────────────────────────────────────────
sys.stdout.write("\n=== plan_single tests ===\n")
pc = MotionGenPlanConfig(enable_graph=True, enable_graph_attempt=10,
                         max_attempts=10, enable_finetune_trajopt=True)
try:
    r = mg.plan_single(js, ee, pc)
    sys.stdout.write(f"Identity plan (graph): success={r.success.item()}, status={r.status}\n")
except Exception as e:
    sys.stdout.write(f"Identity plan (graph): ERROR: {e}\n")
sys.stdout.flush()

# Test with perturbations
for label, delta in [("+X2cm", [0.02,0,0]), ("-X2cm", [-0.02,0,0]), ("+Z2cm", [0,0,0.02]), ("-Z2cm", [0,0,-0.02])]:
    from curobo.types.math import Pose
    tgt = Pose(position=ee.position.clone(), quaternion=ee.quaternion.clone())
    tgt.position[0, 0] += delta[0]; tgt.position[0, 1] += delta[1]; tgt.position[0, 2] += delta[2]
    try:
        r = mg.plan_single(js, tgt, pc)
        sys.stdout.write(f"  {label}: success={r.success.item()}, status={r.status}\n")
    except Exception as e:
        sys.stdout.write(f"  {label}: ERROR: {e}\n")
sys.stdout.flush()

import shutil; shutil.rmtree(tmp_dir, ignore_errors=True)
simulation_app.close()
