#!/usr/bin/env python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terminal-based calibration for XRoboToolkit VR-to-ROS coordinate mapping.

Estimates W_T_Q (OpenXR Quest frame to ROS world frame) and R_rot_map
(controller rotation axis to TCP rotation axis) via multi-point SVD.
Outputs a JSON file consumable by XRoboToolkitDeviceCfg.calibration_json.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Calibrate XRoboToolkit VR-to-ROS coordinate mapping via multi-point SVD."
)
parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Output JSON path (default: logs/piper_calibration/calibration_<timestamp>.json).",
)
parser.add_argument(
    "--tcp_offset_z",
    type=float,
    default=0.13503,
    help="TCP offset from link6 along Z in meters (default: 0.13503).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.piper import PIPER_STANDARD_WITH_GRIPPER_CFG

from xrobotoolkit_teleop.hardware.piper_world_frame_mapping import (
    OPENXR_TO_ROS,
    estimate_rotation_direction_map,
    estimate_W_T_Q,
    make_T,
    openxr_anchor_W_T_Q,
    project_to_so3,
    reprojection_error_stats,
    rotation_direction_error_stats,
    so3_log,
    validate_T,
)


def _create_xr_client():
    """Lazily create the XR client so the script imports cleanly without xrobotoolkit_sdk."""
    try:
        from xrobotoolkit_teleop.common.xr_client import XrClient
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Calibration requires xrobotoolkit_sdk. "
            "In the Docker container, run: "
            "bash /workspace/isaaclab/scripts/tools/setup_xrobotoolkit_env.sh"
        ) from exc

    return XrClient()

_CALIB_POINTS = [
    ("tcp_origin", np.array([0.0, 0.0, 0.0])),
    ("x_plus", np.array([0.12, 0.0, 0.0])),
    ("x_minus", np.array([-0.12, 0.0, 0.0])),
    ("y_plus", np.array([0.0, 0.12, 0.0])),
    ("y_minus", np.array([0.0, -0.12, 0.0])),
    ("z_plus", np.array([0.0, 0.0, 0.10])),
]

_ROTATION_PROMPTS = [
    ("roll", "+X", np.array([1.0, 0.0, 0.0]), "绕 ROS +X 轴旋转 (手腕内翻/外翻)"),
    ("pitch", "-Y", np.array([0.0, -1.0, 0.0]), "绕 ROS -Y 轴旋转 (手腕上弯/下弯)"),
    ("yaw", "+Z", np.array([0.0, 0.0, 1.0]), "绕 ROS +Z 轴旋转 (手腕左转/右转)"),
]

_QUALITY_TRANSLATION_MEAN_M = 0.03
_QUALITY_TRANSLATION_MAX_M = 0.08
_QUALITY_ROTATION_MEAN_DEG = 25.0
_QUALITY_ROTATION_MAX_DEG = 40.0
_QUALITY_ROTATION_MIN_ANGLE_DEG = 15.0
_QUALITY_ROTATION_MAX_ANGLE_DEG = 150.0


@configclass
class CalibSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg()
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    robot = PIPER_STANDARD_WITH_GRIPPER_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )
    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/arm_base",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/link6",
                name="end_effector",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.13503]),
            ),
        ],
    )


def _w_T_tcp(scene: InteractiveScene) -> np.ndarray:
    """Return the current TCP pose as a 4x4 matrix in the world frame."""
    pos = scene["ee_frame"].data.target_pos_w[0, 0].detach().cpu().numpy()
    quat = scene["ee_frame"].data.target_quat_w[0, 0].detach().cpu().numpy()
    # quat is wxyz; convert to xyzw for make_T / ROS convention
    T = make_T()
    T[:3, 3] = pos
    T[:3, :3] = _quat_wxyz_to_R(quat)
    return T


def _quat_wxyz_to_R(q: np.ndarray) -> np.ndarray:
    """Convert wxyz quaternion to 3x3 rotation matrix."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=float,
    )


def _q_T_h(xr_client: XrClient) -> np.ndarray | None:
    """Return the right controller pose as a 4x4 in OpenXR Quest frame."""
    pose = xr_client.get_pose_by_name("right_controller")
    if pose is None:
        return None
    pose = np.asarray(pose, dtype=float)
    if pose.shape[0] < 7 or not np.all(np.isfinite(pose)):
        return None
    q = pose[3:7] / np.linalg.norm(pose[3:7])
    T = make_T()
    T[:3, 3] = pose[:3]
    T[:3, :3] = _quat_wxyz_to_R(np.array([q[3], q[0], q[1], q[2]]))
    return T


def _wait_for_grip(xr_client: XrClient, threshold: float = 0.9) -> np.ndarray:
    """Block until right_grip > threshold, then return Q_T_H."""
    print("    按住 right_grip 以捕获...", flush=True)
    while simulation_app.is_running():
        if float(xr_client.get_key_value_by_name("right_grip")) >= threshold:
            T = _q_T_h(xr_client)
            if T is not None:
                return T
        time.sleep(0.05)
    sys.exit(0)


def _wait_for_button_a(xr_client: XrClient) -> None:
    """Block until A button is pressed and released (edge trigger)."""
    print("    按 A 确认...", flush=True)
    # wait for press
    while simulation_app.is_running():
        if bool(xr_client.get_button_state_by_name("A")):
            break
        time.sleep(0.05)
    # wait for release
    while simulation_app.is_running():
        if not bool(xr_client.get_button_state_by_name("A")):
            break
        time.sleep(0.05)


def _step_sim(sim: sim_utils.SimulationContext, scene: InteractiveScene, sim_dt: float, robot_default_joints: torch.Tensor) -> None:
    """Step the simulation once."""
    scene["robot"].set_joint_position_target(robot_default_joints)
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim_dt)


def run_calibration():
    """Run the interactive calibration pipeline."""
    # ---- Phase 0: Setup ----
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.5, 1.5, 1.5], target=[0.4, 0.0, 0.3])

    scene_cfg = CalibSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    sim_dt = sim.get_physics_dt()
    robot_default_joints = scene["robot"].data.default_joint_pos.clone()

    # Step a few times to settle
    for _ in range(60):
        _step_sim(sim, scene, sim_dt, robot_default_joints)

    W_T_TCP0 = _w_T_tcp(scene)
    print(f"\nTCP 世界坐标 (wrt arm_base):\n  pos = {W_T_TCP0[:3, 3].round(4)} m", flush=True)

    calib_targets = [
        (name, W_T_TCP0[:3, 3] + offset) for name, offset in _CALIB_POINTS
    ]
    print("标定目标点 (世界坐标):")
    for name, xyz in calib_targets:
        print(f"  {name:12s}: {np.array2string(xyz, precision=4, suppress_small=True)}", flush=True)

    # Connect XR client
    print("\n正在连接 XRoboToolkit XR 服务...", flush=True)
    xr_client = _create_xr_client()
    print("XR 服务已连接。\n", flush=True)

    # ---- Phase 1: Single-point anchor ----
    print("=" * 50)
    print("Phase 1: 单点锚定")
    print("将 VR 手柄放在机器人 TCP 位置，按住 right_grip", flush=True)
    Q_T_H_anchor = _wait_for_grip(xr_client)
    W_T_Q = openxr_anchor_W_T_Q(Q_T_H_anchor, W_T_TCP0)
    print(f"  初始 W_T_Q 已计算 (旋转=OPENXR_TO_ROS)\n", flush=True)

    # ---- Phase 2: Multi-point sampling ----
    print("=" * 50)
    print("Phase 2: 多点采样")
    print(f"共 {len(calib_targets)} 个目标点。将手柄移到目标位置后按 A 确认。", flush=True)

    world_xyz_m_list = []
    quest_xyz_m_list = []

    for idx, (name, target_xyz) in enumerate(calib_targets):
        print(f"\n--- 目标 {idx + 1}/{len(calib_targets)}: {name} ---", flush=True)
        print(f"  世界坐标: {np.array2string(target_xyz, precision=4, suppress_small=True)}", flush=True)
        print(f"  偏移 TCP: {np.array2string(_CALIB_POINTS[idx][1], precision=3, suppress_small=True)} m", flush=True)
        _wait_for_button_a(xr_client)

        T = _q_T_h(xr_client)
        if T is None:
            print("  [WARN] 手柄姿态无效，跳过此点", flush=True)
            continue

        world_xyz_m_list.append(target_xyz)
        quest_xyz_m_list.append(T[:3, 3])
        print(f"  已采样: quest_xyz = {np.array2string(T[:3, 3], precision=4, suppress_small=True)}", flush=True)

        # Re-estimate after every sample if enough points
        n = len(world_xyz_m_list)
        if n >= 3:
            world_arr = np.array(world_xyz_m_list)
            quest_arr = np.array(quest_xyz_m_list)
            # Check for collinearity: centroid-subtracted points should span >= 2 dims
            quest_centered = quest_arr - np.mean(quest_arr, axis=0)
            if np.linalg.matrix_rank(quest_centered) >= 2:
                W_T_Q_candidate = estimate_W_T_Q(world_arr, quest_arr)
                errors = reprojection_error_stats(W_T_Q_candidate, world_arr, quest_arr)
                print(f"  当前 W_T_Q reprojection: mean={errors['mean_m']:.4f}m, max={errors['max_m']:.4f}m", flush=True)
            else:
                print("  [WARN] 采样点共线, 跳过本次估计", flush=True)

    if len(world_xyz_m_list) < 3:
        print("\n[FAIL] 采样点不足 (< 3), 无法估计 W_T_Q", flush=True)
        sys.exit(1)

    world_arr = np.array(world_xyz_m_list)
    quest_arr = np.array(quest_xyz_m_list)
    W_T_Q = estimate_W_T_Q(world_arr, quest_arr)
    errors = reprojection_error_stats(W_T_Q, world_arr, quest_arr)

    print(f"\n--- 平移标定结果 ---", flush=True)
    print(f"  W_T_Q rotation:\n{np.array2string(W_T_Q[:3, :3], precision=4, suppress_small=True)}", flush=True)
    print(f"  mean_error = {errors['mean_m']:.4f} m", flush=True)
    print(f"  max_error  = {errors['max_m']:.4f} m", flush=True)

    translation_ok = errors["mean_m"] <= _QUALITY_TRANSLATION_MEAN_M and errors["max_m"] <= _QUALITY_TRANSLATION_MAX_M
    if not translation_ok:
        print(
            f"\n[WARN] 平移标定未达标 (mean<={_QUALITY_TRANSLATION_MEAN_M}m, max<={_QUALITY_TRANSLATION_MAX_M}m). "
            "建议重新标定。",
            flush=True,
        )

    # ---- Phase 3: Rotation direction calibration ----
    print("\n" + "=" * 50)
    print("Phase 3: 旋转方向标定")
    print("手柄放在 TCP 位置附近, 做指定旋转动作", flush=True)

    controller_axes_list = []
    target_axes_list = []
    angles_deg_list = []

    for rot_name, axis_label, target_axis, prompt in _ROTATION_PROMPTS:
        print(f"\n--- 旋转: {rot_name} ({axis_label}) ---", flush=True)
        print(f"  {prompt}", flush=True)
        print("  先保持中性姿态，按 A 捕获...", flush=True)
        _wait_for_button_a(xr_client)
        T_neutral = _q_T_h(xr_client)
        if T_neutral is None:
            print("  [WARN] 中性姿态无效，跳过", flush=True)
            continue

        print("  现在做旋转动作，保持旋转后姿态，按 A 捕获...", flush=True)
        _wait_for_button_a(xr_client)
        T_rotated = _q_T_h(xr_client)
        if T_rotated is None:
            print("  [WARN] 旋转姿态无效，跳过", flush=True)
            continue

        R_delta = project_to_so3(T_rotated[:3, :3] @ T_neutral[:3, :3].T)
        rotvec_controller = so3_log(R_delta)
        angle_deg = float(np.rad2deg(np.linalg.norm(rotvec_controller)))

        print(f"  控制器旋转角度: {angle_deg:.1f} deg", flush=True)

        if angle_deg < _QUALITY_ROTATION_MIN_ANGLE_DEG:
            print(f"  [WARN] 旋转角度过小 (< {_QUALITY_ROTATION_MIN_ANGLE_DEG} deg), 跳过", flush=True)
            continue
        if angle_deg > _QUALITY_ROTATION_MAX_ANGLE_DEG:
            print(f"  [WARN] 旋转角度过大 (> {_QUALITY_ROTATION_MAX_ANGLE_DEG} deg), 跳过", flush=True)
            continue

        R_W_Q = project_to_so3(W_T_Q[:3, :3])
        rotvec_world = R_W_Q @ rotvec_controller
        controller_axis = rotvec_world / np.linalg.norm(rotvec_world)

        controller_axes_list.append(controller_axis)
        target_axes_list.append(target_axis)
        angles_deg_list.append(angle_deg)
        print(f"  控制器轴(world): {np.array2string(controller_axis, precision=3, suppress_small=True)}", flush=True)

    if len(controller_axes_list) < 2:
        print("\n[FAIL] 旋转样本不足 (< 2), 使用 identity R_rot_map", flush=True)
        R_rot_map = np.eye(3, dtype=float)
    else:
        controller_axes_arr = np.array(controller_axes_list)
        target_axes_arr = np.array(target_axes_list)
        R_rot_map = estimate_rotation_direction_map(controller_axes_arr, target_axes_arr)
        rot_errors = rotation_direction_error_stats(controller_axes_arr, target_axes_arr, R_rot_map)

        print(f"\n--- 旋转标定结果 ---", flush=True)
        print(f"  R_rot_map:\n{np.array2string(R_rot_map, precision=4, suppress_small=True)}", flush=True)
        print(f"  mean_axis_error = {rot_errors['mean_axis_error_deg']:.2f} deg", flush=True)
        print(f"  max_axis_error  = {rot_errors['max_axis_error_deg']:.2f} deg", flush=True)
        print(f"  per_sample: {np.array2string(rot_errors['per_sample_deg'], precision=1, suppress_small=True)} deg", flush=True)

        rotation_ok = (
            rot_errors["mean_axis_error_deg"] <= _QUALITY_ROTATION_MEAN_DEG
            and rot_errors["max_axis_error_deg"] <= _QUALITY_ROTATION_MAX_DEG
        )
        if not rotation_ok:
            print(
                f"\n[WARN] 旋转标定未达标 "
                f"(mean<={_QUALITY_ROTATION_MEAN_DEG}deg, max<={_QUALITY_ROTATION_MAX_DEG}deg). "
                "建议重新标定。",
                flush=True,
            )

    # ---- Phase 4: Output ----
    accepted = translation_ok and (
        len(controller_axes_list) < 2
        or (
            rotation_direction_error_stats(
                np.array(controller_axes_list), np.array(target_axes_list), R_rot_map
            )["mean_axis_error_deg"]
            <= _QUALITY_ROTATION_MEAN_DEG
            and rotation_direction_error_stats(
                np.array(controller_axes_list), np.array(target_axes_list), R_rot_map
            )["max_axis_error_deg"]
            <= _QUALITY_ROTATION_MAX_DEG
        )
    )

    calib_data = {
        "accepted": bool(accepted),
        "W_T_Q": W_T_Q.tolist(),
        "R_align": np.eye(3, dtype=float).tolist(),
        "R_rot_map": R_rot_map.tolist(),
        "R_align_rpy_rad": [0.0, 0.0, 0.0],
        "rotation_direction_calibration": {
            "controller_axes": [a.tolist() for a in controller_axes_list],
            "target_axes": [a.tolist() for a in target_axes_list],
            "angles_deg": angles_deg_list,
        },
        "translation_reprojection_error": errors,
        "samples": [
            {"name": _CALIB_POINTS[i][0], "world_xyz_m": w.tolist(), "quest_xyz_m": q.tolist()}
            for i, (w, q) in enumerate(zip(world_xyz_m_list, quest_xyz_m_list))
            if i < len(_CALIB_POINTS)
        ],
    }

    if args_cli.output:
        output_path = args_cli.output
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(os.path.dirname(__file__), "..", "..", "logs", "piper_calibration")
        os.makedirs(log_dir, exist_ok=True)
        output_path = os.path.join(log_dir, f"calibration_{timestamp}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(calib_data, f, indent=2)

    print("\n" + "=" * 50)
    print(f"标定 JSON 已写入: {output_path}", flush=True)
    status = "ACCEPTED" if accepted else "REJECTED (质量未达标)"
    print(f"状态: {status}", flush=True)

    if accepted:
        print("\n使用方法:", flush=True)
        print(f"  teleop_se3_agent.py --xrobotoolkit_calibration_json {output_path}", flush=True)
        print(f"  record_demos.py --xrobotoolkit_calibration_json {output_path}", flush=True)

    return calib_data


def main():
    run_calibration()
    simulation_app.close()


if __name__ == "__main__":
    main()
