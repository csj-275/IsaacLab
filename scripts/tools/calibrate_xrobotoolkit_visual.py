#!/usr/bin/env python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Sim viewport calibration for XRoboToolkit world-frame teleoperation.

The script estimates the OpenXR tracking frame to Piper world-frame mapping and
the controller-rotation direction map used by XRoboToolkitDevice calibration
JSON files. It visualizes the calibration points and frames in Isaac Sim and
does not connect to hardware or send robot commands.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

from isaaclab.app import AppLauncher


DEFAULT_TASK_NAME = "Isaac-Stack-Cube-Piper-IK-Rel-v0"
DEFAULT_POSE_SOURCE = "right_controller"
DEFAULT_ANCHOR_KEY = "right_grip"
DEFAULT_CAPTURE_KEY = "A"
DEFAULT_RETRY_KEY = "B"
DEFAULT_EXIT_BUTTON = "right_axis_click"
DEFAULT_LOG_DIR = "logs/piper_calibration"
DEFAULT_RATE_HZ = 30.0
DEFAULT_SAMPLE_COUNT = 15
DEFAULT_ANCHOR_THRESHOLD = 0.9
DEFAULT_RELATIVE_STEP_M = 0.12
DEFAULT_RELATIVE_LIFT_M = 0.10
DEFAULT_MIN_POINT_COUNT = 6
DEFAULT_MAX_MEAN_ERROR_M = 0.03
DEFAULT_MAX_MAX_ERROR_M = 0.08
DEFAULT_ROTATION_MIN_ANGLE_DEG = 15.0
DEFAULT_ROTATION_MAX_ANGLE_DEG = 150.0
DEFAULT_MAX_ROTATION_MEAN_AXIS_ERROR_DEG = 25.0
DEFAULT_MAX_ROTATION_MAX_AXIS_ERROR_DEG = 40.0
DEFAULT_PREVIEW_AFTER_FIT = True


parser = argparse.ArgumentParser(
    description=(
        "Calibrate XRoboToolkit world-frame mapping in an Isaac Sim viewport. "
        "This does not connect to Piper hardware or send CAN commands."
    )
)
parser.add_argument("--output", type=str, default=None, help="Output calibration JSON path.")
parser.add_argument("--log-dir", type=str, default=DEFAULT_LOG_DIR, help="Output directory when --output is omitted.")
parser.add_argument("--task", type=str, default=DEFAULT_TASK_NAME, help="Task name printed in ready-to-run commands.")
parser.add_argument("--pose-source", type=str, default=DEFAULT_POSE_SOURCE, help="XRoboToolkit pose source name.")
parser.add_argument("--anchor-key", type=str, default=DEFAULT_ANCHOR_KEY, help="Key used for preview anchoring.")
parser.add_argument("--anchor-threshold", type=float, default=DEFAULT_ANCHOR_THRESHOLD)
parser.add_argument("--capture-key", type=str, default=DEFAULT_CAPTURE_KEY, help="Key/button used to record samples.")
parser.add_argument("--retry-key", type=str, default=DEFAULT_RETRY_KEY, help="Key/button used to redo the current sample.")
parser.add_argument("--exit-button", type=str, default=DEFAULT_EXIT_BUTTON, help="Button used to exit calibration.")
parser.add_argument("--rate-hz", type=float, default=DEFAULT_RATE_HZ, help="XR polling and viewport update rate in Hz.")
parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT, help="Number of XR poses averaged per capture.")
parser.add_argument("--relative-step-m", type=float, default=DEFAULT_RELATIVE_STEP_M)
parser.add_argument("--relative-lift-m", type=float, default=DEFAULT_RELATIVE_LIFT_M)
parser.add_argument("--min-point-count", type=int, default=DEFAULT_MIN_POINT_COUNT)
parser.add_argument("--max-mean-error-m", type=float, default=DEFAULT_MAX_MEAN_ERROR_M)
parser.add_argument("--max-max-error-m", type=float, default=DEFAULT_MAX_MAX_ERROR_M)
parser.add_argument("--rotation-min-angle-deg", type=float, default=DEFAULT_ROTATION_MIN_ANGLE_DEG)
parser.add_argument("--rotation-max-angle-deg", type=float, default=DEFAULT_ROTATION_MAX_ANGLE_DEG)
parser.add_argument("--max-rotation-mean-axis-error-deg", type=float, default=DEFAULT_MAX_ROTATION_MEAN_AXIS_ERROR_DEG)
parser.add_argument("--max-rotation-max-axis-error-deg", type=float, default=DEFAULT_MAX_ROTATION_MAX_AXIS_ERROR_DEG)
parser.add_argument(
    "--preview-after-fit",
    action=argparse.BooleanOptionalAction,
    default=DEFAULT_PREVIEW_AFTER_FIT,
    help="Keep updating the fitted TCP target in the viewport after writing the JSON.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.piper import PIPER_STANDARD_WITH_GRIPPER_CFG
from xrobotoolkit_teleop.hardware.piper_world_frame_mapping import (
    OPENXR_TO_ROS,
    estimate_W_T_Q,
    estimate_rotation_direction_map,
    make_T,
    openxr_anchor_W_T_Q,
    openxr_to_ros_tracking_T,
    project_to_so3,
    reprojection_error_stats,
    rotation_direction_error_stats,
    rotation_quality,
    so3_log,
    world_frame_calibrated_target,
)


ROTATION_DIRECTION_ACTIONS = (
    ("roll", np.array([1.0, 0.0, 0.0], dtype=float), "做明显翻转动作，目标轴为 ROS +X。"),
    ("pitch_up", np.array([0.0, -1.0, 0.0], dtype=float), "让手柄前端向上抬，目标轴为 ROS -Y。"),
    ("yaw_left", np.array([0.0, 0.0, 1.0], dtype=float), "让手柄前端向左摆，目标轴为 ROS +Z。"),
)


@configclass
class CalibSceneCfg(InteractiveSceneCfg):
    """Minimal Piper scene used only for viewport calibration."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    robot = PIPER_STANDARD_WITH_GRIPPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/arm_base",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/link6",
                name="end_effector",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.13503]),
            )
        ],
    )


@dataclass
class CalibrationVisualizers:
    """Viewport marker handles used by the calibration loop."""

    frames: VisualizationMarkers
    points: VisualizationMarkers


def _create_xr_client():
    try:
        from xrobotoolkit_teleop.common.xr_client import XrClient
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Calibration requires xrobotoolkit_sdk and xrobotoolkit_teleop. "
            "In the Docker container, run: bash /workspace/isaaclab/scripts/tools/setup_xrobotoolkit_env.sh"
        ) from exc

    return XrClient()


def _make_visualizers() -> CalibrationVisualizers:
    frame_cfg = FRAME_MARKER_CFG.copy()
    frame_cfg.prim_path = "/Visuals/XRoboToolkitCalibrationFrames"
    frame_cfg.markers["frame"].scale = (0.12, 0.12, 0.12)

    point_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/XRoboToolkitCalibrationPoints",
        markers={
            "target": sim_utils.SphereCfg(
                radius=0.016,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.3, 1.0)),
            ),
            "active": sim_utils.SphereCfg(
                radius=0.028,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.85, 0.25)),
            ),
            "sample": sim_utils.SphereCfg(
                radius=0.014,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.45, 0.0)),
            ),
            "anchor": sim_utils.SphereCfg(
                radius=0.022,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.75, 0.55)),
            ),
        },
    )
    return CalibrationVisualizers(frames=VisualizationMarkers(frame_cfg), points=VisualizationMarkers(point_cfg))


def _quat_xyzw_to_R(quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = [float(v) for v in np.asarray(quat_xyzw, dtype=float)]
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=float,
    )


def _R_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    R = project_to_so3(R)
    trace = float(np.trace(R))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=float)
    return quat / max(float(np.linalg.norm(quat)), 1.0e-9)


def _pose_to_T(pose_xyzw: np.ndarray) -> np.ndarray:
    pose_xyzw = np.asarray(pose_xyzw, dtype=float)
    quat = pose_xyzw[3:7]
    quat = quat / max(float(np.linalg.norm(quat)), 1.0e-9)
    T = make_T(_quat_xyzw_to_R(quat), pose_xyzw[:3])
    return T


def _T_to_pos_quat_wxyz(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T = np.asarray(T, dtype=float)
    return T[:3, 3].copy(), _R_to_quat_wxyz(T[:3, :3])


def _average_poses(poses: list[np.ndarray]) -> np.ndarray:
    stacked = np.stack(poses, axis=0)
    averaged = np.mean(stacked, axis=0)
    quats = stacked[:, 3:7].copy()
    ref = quats[0].copy()
    for index in range(quats.shape[0]):
        if float(np.dot(quats[index], ref)) < 0.0:
            quats[index] = -quats[index]
    quat = np.mean(quats, axis=0)
    averaged[3:7] = quat / max(float(np.linalg.norm(quat)), 1.0e-9)
    return averaged


def _w_T_tcp(scene: InteractiveScene) -> np.ndarray:
    pos = scene["ee_frame"].data.target_pos_w[0, 0].detach().cpu().numpy()
    quat_wxyz = scene["ee_frame"].data.target_quat_w[0, 0].detach().cpu().numpy()
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=float)
    return make_T(_quat_xyzw_to_R(quat_xyzw), pos)


def _read_pose(xr_client: Any, pose_source: str) -> np.ndarray | None:
    pose = xr_client.get_pose_by_name(pose_source)
    if pose is None:
        return None
    pose = np.asarray(pose, dtype=float)
    if pose.shape[0] < 7 or not np.all(np.isfinite(pose[:7])):
        return None
    quat = pose[3:7]
    quat_norm = float(np.linalg.norm(quat))
    if quat_norm < 1.0e-9:
        return None
    pose = pose[:7].copy()
    pose[3:7] = quat / quat_norm
    return pose


def _key_active(xr_client: Any, key_name: str, threshold: float = 0.9) -> bool:
    try:
        return float(xr_client.get_key_value_by_name(key_name)) >= float(threshold)
    except Exception:
        return bool(xr_client.get_button_state_by_name(key_name))


def _button_pressed(xr_client: Any, button_name: str) -> bool:
    try:
        return bool(xr_client.get_button_state_by_name(button_name))
    except Exception:
        return _key_active(xr_client, button_name, 0.9)


def _step_sim(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    sim_dt: float,
    robot_default_joints: torch.Tensor,
) -> None:
    scene["robot"].set_joint_position_target(robot_default_joints)
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim_dt)


def _capture_average_pose(
    xr_client: Any,
    pose_source: str,
    rate_hz: float,
    sample_count: int,
    step_callback,
) -> np.ndarray | None:
    poses: list[np.ndarray] = []
    dt = 1.0 / max(rate_hz, 1.0)
    for _ in range(max(sample_count, 1)):
        pose = _read_pose(xr_client, pose_source)
        if pose is not None:
            poses.append(pose)
        step_callback()
        time.sleep(dt)
    if not poses:
        return None
    return _average_poses(poses)


def _wait_for_anchor(
    xr_client: Any,
    pose_source: str,
    anchor_key: str,
    anchor_threshold: float,
    exit_button: str,
    rate_hz: float,
    sample_count: int,
    draw_callback,
) -> np.ndarray | None:
    previous_anchor = _key_active(xr_client, anchor_key, anchor_threshold)
    dt = 1.0 / max(rate_hz, 1.0)
    while simulation_app.is_running():
        if _button_pressed(xr_client, exit_button):
            return None
        draw_callback()
        anchor = _key_active(xr_client, anchor_key, anchor_threshold)
        if anchor and not previous_anchor:
            return _capture_average_pose(xr_client, pose_source, rate_hz, sample_count, draw_callback)
        previous_anchor = anchor
        time.sleep(dt)
    return None


def _wait_for_capture_or_retry(
    xr_client: Any,
    pose_source: str,
    capture_key: str,
    retry_key: str,
    exit_button: str,
    rate_hz: float,
    sample_count: int,
    draw_callback,
) -> tuple[str, np.ndarray | None]:
    previous_capture = _key_active(xr_client, capture_key, 0.9)
    previous_retry = _key_active(xr_client, retry_key, 0.9)
    dt = 1.0 / max(rate_hz, 1.0)
    while simulation_app.is_running():
        if _button_pressed(xr_client, exit_button):
            return "exit", None
        draw_callback()
        retry = _key_active(xr_client, retry_key, 0.9)
        if retry and not previous_retry:
            return "retry", None
        capture = _key_active(xr_client, capture_key, 0.9)
        if capture and not previous_capture:
            pose = _capture_average_pose(xr_client, pose_source, rate_hz, sample_count, draw_callback)
            return "capture", pose
        previous_capture = capture
        previous_retry = retry
        time.sleep(dt)
    return "exit", None


def _relative_calibration_targets(ref_tcp_T: np.ndarray, step_m: float, lift_m: float) -> dict[str, Any]:
    if step_m <= 0.0 or lift_m <= 0.0:
        raise ValueError("relative-step-m and relative-lift-m must be positive.")
    offsets = [
        ("tcp_origin", [0.0, 0.0, 0.0], "TCP 原点"),
        ("x_plus", [step_m, 0.0, 0.0], f"从 TCP 原点沿 ROS +X 前方移动 {step_m:.3f} m"),
        ("x_minus", [-step_m, 0.0, 0.0], f"从 TCP 原点沿 ROS -X 后方移动 {step_m:.3f} m"),
        ("y_plus", [0.0, step_m, 0.0], f"从 TCP 原点沿 ROS +Y 左侧移动 {step_m:.3f} m"),
        ("y_minus", [0.0, -step_m, 0.0], f"从 TCP 原点沿 ROS -Y 右侧移动 {step_m:.3f} m"),
        ("z_plus", [0.0, 0.0, lift_m], f"从 TCP 原点沿 ROS +Z 上方移动 {lift_m:.3f} m"),
    ]
    names = [item[0] for item in offsets]
    descriptions = [item[2] for item in offsets]
    offset_xyz = np.asarray([item[1] for item in offsets], dtype=float)
    origin = np.asarray(ref_tcp_T, dtype=float)[:3, 3]
    return {
        "names": names,
        "descriptions": descriptions,
        "offset_xyz_m": offset_xyz,
        "world_xyz_m": origin[None, :] + offset_xyz,
        "origin_world_xyz_m": origin.copy(),
    }


def _visualize(
    visualizers: CalibrationVisualizers,
    scene: InteractiveScene,
    target_xyz_m: np.ndarray,
    active_index: int | None,
    samples: list[dict[str, Any]],
    W_T_Q: np.ndarray | None,
    R_align: np.ndarray,
    R_rot_map: np.ndarray,
    live_pose: np.ndarray | None,
    ref_Q_T_H: np.ndarray | None,
    ref_tcp_T: np.ndarray,
) -> None:
    frames: list[np.ndarray] = [_w_T_tcp(scene), ref_tcp_T]
    marker_points: list[np.ndarray] = []
    marker_indices: list[int] = []

    for index, xyz in enumerate(target_xyz_m):
        marker_points.append(np.asarray(xyz, dtype=float))
        marker_indices.append(1 if active_index == index else 0)
    marker_points.append(ref_tcp_T[:3, 3].copy())
    marker_indices.append(3)

    if live_pose is not None:
        Q_T_H = _pose_to_T(live_pose)
        frames.append(Q_T_H)
        frames.append(openxr_to_ros_tracking_T(Q_T_H))
        if W_T_Q is not None:
            W_T_H = W_T_Q @ Q_T_H
            W_T_H_aligned = W_T_H.copy()
            W_T_H_aligned[:3, :3] = project_to_so3(W_T_H_aligned[:3, :3] @ R_align)
            frames.append(W_T_H_aligned)
            target_T = ref_tcp_T
            if ref_Q_T_H is not None:
                target_T = world_frame_calibrated_target(
                    W_T_Q,
                    R_align,
                    ref_Q_T_H,
                    Q_T_H,
                    ref_tcp_T,
                    scale=1.0,
                    R_rot_map=R_rot_map,
                )
            frames.append(target_T)

    for sample in samples:
        W_T_H = sample.get("W_T_H")
        if W_T_H is not None:
            marker_points.append(np.asarray(W_T_H, dtype=float)[:3, 3].copy())
            marker_indices.append(2)

    frame_translations = []
    frame_orientations = []
    for frame_T in frames:
        pos, quat = _T_to_pos_quat_wxyz(frame_T)
        frame_translations.append(pos)
        frame_orientations.append(quat)
    visualizers.frames.visualize(
        np.asarray(frame_translations, dtype=np.float32),
        np.asarray(frame_orientations, dtype=np.float32),
        marker_indices=np.zeros(len(frame_translations), dtype=np.int32),
    )
    visualizers.points.visualize(
        np.asarray(marker_points, dtype=np.float32),
        marker_indices=np.asarray(marker_indices, dtype=np.int32),
    )


def _fit_translation(samples: list[dict[str, Any]], target_xyz_m: np.ndarray) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    if len(samples) < 3:
        return None, None
    quest_xyz_m = np.stack([np.asarray(sample["Q_T_H"], dtype=float)[:3, 3] for sample in samples])
    quest_centered = quest_xyz_m - quest_xyz_m.mean(axis=0)
    if np.linalg.matrix_rank(quest_centered, tol=1.0e-6) < 3:
        return None, None
    world_xyz_m = target_xyz_m[: len(samples)]
    W_T_Q = estimate_W_T_Q(world_xyz_m, quest_xyz_m)
    W_T_Q[:3, :3] = project_to_so3(W_T_Q[:3, :3])
    return W_T_Q, reprojection_error_stats(W_T_Q, world_xyz_m, quest_xyz_m)


def _capture_rotation_direction_calibration(
    xr_client: Any,
    visualizers: CalibrationVisualizers,
    scene: InteractiveScene,
    target_xyz_m: np.ndarray,
    samples: list[dict[str, Any]],
    W_T_Q: np.ndarray,
    R_align: np.ndarray,
    ref_Q_T_H: np.ndarray,
    ref_tcp_T: np.ndarray,
    sim: sim_utils.SimulationContext,
    sim_dt: float,
    robot_default_joints: torch.Tensor,
) -> dict[str, Any] | None:
    controller_axes: list[np.ndarray] = []
    target_axes: list[np.ndarray] = []
    actions: list[dict[str, Any]] = []
    failures: list[str] = []

    print("-" * 72)
    print("开始旋转方向校准。每个动作先按 A 记录中立姿态，再做动作并按 A 记录动作姿态。")
    print(f"按 {args_cli.retry_key} 可重做当前姿态采样，按 {args_cli.exit_button} 退出。")

    def draw() -> None:
        _step_sim(sim, scene, sim_dt, robot_default_joints)
        _visualize(
            visualizers,
            scene,
            target_xyz_m,
            None,
            samples,
            W_T_Q,
            R_align,
            np.eye(3, dtype=float),
            _read_pose(xr_client, args_cli.pose_source),
            ref_Q_T_H,
            ref_tcp_T,
        )

    for action_index, (name, target_axis, description) in enumerate(ROTATION_DIRECTION_ACTIONS):
        print("-" * 72)
        print(f"旋转动作 {action_index + 1}/{len(ROTATION_DIRECTION_ACTIONS)}: {name}")
        print(description)

        neutral_pose = None
        while neutral_pose is None:
            print("请回到中立姿态，保持控制器位置基本不变，然后按 A。")
            status, pose = _wait_for_capture_or_retry(
                xr_client,
                args_cli.pose_source,
                args_cli.capture_key,
                args_cli.retry_key,
                args_cli.exit_button,
                args_cli.rate_hz,
                args_cli.sample_count,
                draw,
            )
            if status == "exit":
                return None
            if status == "retry":
                continue
            neutral_pose = pose
            if neutral_pose is None:
                print("未获取到有效中立姿态，重试。")

        action_pose = None
        while action_pose is None:
            print("现在执行该方向的明显旋转动作，保持平移尽量小，然后按 A。")
            status, pose = _wait_for_capture_or_retry(
                xr_client,
                args_cli.pose_source,
                args_cli.capture_key,
                args_cli.retry_key,
                args_cli.exit_button,
                args_cli.rate_hz,
                args_cli.sample_count,
                draw,
            )
            if status == "exit":
                return None
            if status == "retry":
                continue
            action_pose = pose
            if action_pose is None:
                print("未获取到有效动作姿态，重试。")

        neutral_Q_T_H = _pose_to_T(neutral_pose)
        action_Q_T_H = _pose_to_T(action_pose)
        neutral_W_T_H = W_T_Q @ neutral_Q_T_H
        action_W_T_H = W_T_Q @ action_Q_T_H
        R_delta_controller = project_to_so3(action_W_T_H[:3, :3] @ neutral_W_T_H[:3, :3].T)
        rotvec_controller = so3_log(R_delta_controller)
        angle_rad = float(np.linalg.norm(rotvec_controller))
        angle_deg = float(np.rad2deg(angle_rad))
        controller_axis = rotvec_controller / angle_rad if angle_rad > 1.0e-9 else np.zeros(3, dtype=float)

        valid_angle = args_cli.rotation_min_angle_deg <= angle_deg <= args_cli.rotation_max_angle_deg
        action_report = {
            "name": name,
            "description": description,
            "target_axis_W": target_axis.copy(),
            "neutral_pose_xyzw": neutral_pose.copy(),
            "action_pose_xyzw": action_pose.copy(),
            "neutral_Q_T_H": neutral_Q_T_H.copy(),
            "action_Q_T_H": action_Q_T_H.copy(),
            "neutral_W_T_H": neutral_W_T_H.copy(),
            "action_W_T_H": action_W_T_H.copy(),
            "R_delta_controller": R_delta_controller.copy(),
            "rotvec_controller": rotvec_controller.copy(),
            "angle_deg": angle_deg,
            "controller_axis_W": controller_axis.copy(),
            "valid_angle": bool(valid_angle),
            "axis_error_deg": None,
        }
        actions.append(action_report)
        print(
            f"记录 {name}: angle={angle_deg:.2f} deg, "
            f"axis={np.array2string(controller_axis, precision=4, suppress_small=True)}"
        )
        if angle_rad <= 1.0e-9:
            failures.append(f"{name}: rotation angle is too small to estimate an axis")
        elif not valid_angle:
            failures.append(
                f"{name}: rotation angle {angle_deg:.2f} deg outside "
                f"[{args_cli.rotation_min_angle_deg:.2f}, {args_cli.rotation_max_angle_deg:.2f}] deg"
            )
        else:
            controller_axes.append(controller_axis.copy())
            target_axes.append(target_axis.copy())

    R_rot_map = np.eye(3, dtype=float)
    error_stats = {"per_sample_deg": np.array([], dtype=float), "mean_axis_error_deg": None, "max_axis_error_deg": None}
    if len(controller_axes) < 2:
        failures.append(f"rotation direction samples {len(controller_axes)} < 2")
    else:
        controller_axes_array = np.stack(controller_axes, axis=0)
        target_axes_array = np.stack(target_axes, axis=0)
        R_rot_map = estimate_rotation_direction_map(controller_axes_array, target_axes_array)
        error_stats = rotation_direction_error_stats(controller_axes_array, target_axes_array, R_rot_map)
        included_index = 0
        for action_report in actions:
            if action_report["valid_angle"]:
                action_report["axis_error_deg"] = float(error_stats["per_sample_deg"][included_index])
                included_index += 1
        if error_stats["mean_axis_error_deg"] > args_cli.max_rotation_mean_axis_error_deg:
            failures.append(
                f"mean rotation axis error {error_stats['mean_axis_error_deg']:.2f} deg > "
                f"{args_cli.max_rotation_mean_axis_error_deg:.2f} deg"
            )
        if error_stats["max_axis_error_deg"] > args_cli.max_rotation_max_axis_error_deg:
            failures.append(
                f"max rotation axis error {error_stats['max_axis_error_deg']:.2f} deg > "
                f"{args_cli.max_rotation_max_axis_error_deg:.2f} deg"
            )

    return {
        "enabled": True,
        "accepted": not failures,
        "R_rot_map": R_rot_map,
        "actions": actions,
        "failures": failures,
        "mean_axis_error_deg": error_stats["mean_axis_error_deg"],
        "max_axis_error_deg": error_stats["max_axis_error_deg"],
        "thresholds": {
            "min_angle_deg": float(args_cli.rotation_min_angle_deg),
            "max_angle_deg": float(args_cli.rotation_max_angle_deg),
            "max_mean_axis_error_deg": float(args_cli.max_rotation_mean_axis_error_deg),
            "max_max_axis_error_deg": float(args_cli.max_rotation_max_axis_error_deg),
        },
    }


def _teleop_command(output_path: str) -> str:
    return (
        "TERM=xterm ./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py "
        f"--task {args_cli.task} --teleop_device xrobotoolkit "
        "--xrobotoolkit_mapping_mode world_frame_calibrated "
        f"--xrobotoolkit_calibration_json {output_path}"
    )


def _recording_command(output_path: str) -> str:
    return (
        "TERM=xterm ./isaaclab.sh -p scripts/tools/record_demos.py "
        f"--task {args_cli.task} --teleop_device xrobotoolkit "
        "--xrobotoolkit_mapping_mode world_frame_calibrated "
        f"--xrobotoolkit_calibration_json {output_path} "
        "--dataset_file ./datasets/piper_xrobo_demo.hdf5"
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def run_calibration() -> str | None:
    os.makedirs(args_cli.log_dir, exist_ok=True)

    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.2, 1.0, 0.8], target=[0.35, 0.0, 0.25])
    scene = InteractiveScene(CalibSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    sim_dt = sim.get_physics_dt()
    robot_default_joints = scene["robot"].data.default_joint_pos.clone()
    visualizers = _make_visualizers()

    for _ in range(60):
        _step_sim(sim, scene, sim_dt, robot_default_joints)

    ref_tcp_T = _w_T_tcp(scene)
    target_config = _relative_calibration_targets(ref_tcp_T, args_cli.relative_step_m, args_cli.relative_lift_m)
    target_names = target_config["names"]
    target_descriptions = target_config["descriptions"]
    target_xyz_m = target_config["world_xyz_m"]
    target_offset_xyz_m = target_config["offset_xyz_m"]

    print("=" * 72)
    print("XRoboToolkit IsaacLab visual world-frame calibration")
    print("=" * 72)
    print("安全说明: 本脚本只运行 Isaac Sim 场景，不连接 Piper SDK，不发送 CAN / move_j / gripper 命令。")
    print("OpenXR -> ROS 基础轴映射: ROS x=-OpenXR z, ROS y=-OpenXR x, ROS z=OpenXR y。")
    print(f"TCP world position: {np.array2string(ref_tcp_T[:3, 3], precision=4, suppress_small=True)} m")
    print(f"预锚定: {args_cli.anchor_key}; 记录样本: {args_cli.capture_key}; 重做当前样本: {args_cli.retry_key}; 退出: {args_cli.exit_button}")
    print("正在初始化 XR SDK...")

    xr_client = _create_xr_client()
    R_align = np.eye(3, dtype=float)
    R_rot_map = np.eye(3, dtype=float)
    W_T_Q: np.ndarray | None = None
    ref_Q_T_H: np.ndarray | None = None
    anchor_W_T_Q: np.ndarray | None = None
    try:
        def draw(active_index: int | None, samples: list[dict[str, Any]]) -> None:
            _step_sim(sim, scene, sim_dt, robot_default_joints)
            _visualize(
                visualizers,
                scene,
                target_xyz_m,
                active_index,
                samples,
                W_T_Q,
                R_align,
                R_rot_map,
                _read_pose(xr_client, args_cli.pose_source),
                ref_Q_T_H,
                ref_tcp_T,
            )

        print("-" * 72)
        print("等待预锚定。把控制器放到舒适初始姿态，按下 right_grip。")
        anchor_pose = _wait_for_anchor(
            xr_client,
            args_cli.pose_source,
            args_cli.anchor_key,
            args_cli.anchor_threshold,
            args_cli.exit_button,
            args_cli.rate_hz,
            args_cli.sample_count,
            lambda: draw(None, []),
        )
        if anchor_pose is None:
            print("校准已中止。")
            return None
        ref_Q_T_H = _pose_to_T(anchor_pose)
        anchor_W_T_Q = openxr_anchor_W_T_Q(ref_Q_T_H, ref_tcp_T)
        anchor_W_T_Q[:3, :3] = project_to_so3(anchor_W_T_Q[:3, :3])
        W_T_Q = anchor_W_T_Q.copy()
        anchor_position_error_m = float(np.linalg.norm((W_T_Q @ ref_Q_T_H)[:3, 3] - ref_tcp_T[:3, 3]))

        print("预锚定完成。开始多点平移采样。")
        samples: list[dict[str, Any]] = []
        for index, (name, world_xyz, offset_xyz, description) in enumerate(
            zip(target_names, target_xyz_m, target_offset_xyz_m, target_descriptions)
        ):
            while True:
                print("-" * 72)
                print(f"样本 {index + 1}/{len(target_names)}: {name}")
                print(description)
                print(f"offset_xyz_m = {np.array2string(offset_xyz, precision=4, suppress_small=True)}")
                print(f"world_xyz_m  = {np.array2string(world_xyz, precision=4, suppress_small=True)}")
                print(f"移动控制器到该位置后按 {args_cli.capture_key}；按 {args_cli.retry_key} 重做当前样本。")
                status, pose = _wait_for_capture_or_retry(
                    xr_client,
                    args_cli.pose_source,
                    args_cli.capture_key,
                    args_cli.retry_key,
                    args_cli.exit_button,
                    args_cli.rate_hz,
                    args_cli.sample_count,
                    lambda index=index: draw(index, samples),
                )
                if status == "exit":
                    print("校准已中止。")
                    return None
                if status == "retry":
                    continue
                if pose is None:
                    print("未获取到有效姿态，重试当前样本。")
                    continue
                Q_T_H = _pose_to_T(pose)
                sample = {
                    "name": name,
                    "world_xyz_m": np.asarray(world_xyz, dtype=float).copy(),
                    "offset_xyz_m": np.asarray(offset_xyz, dtype=float).copy(),
                    "pose_xyzw": pose.copy(),
                    "Q_T_H": Q_T_H.copy(),
                }
                samples.append(sample)
                trial_W_T_Q, trial_errors = _fit_translation(samples, target_xyz_m)
                if trial_W_T_Q is not None:
                    W_T_Q = trial_W_T_Q
                    print(
                        f"当前拟合: mean={trial_errors['mean_m']:.4f} m, "
                        f"max={trial_errors['max_m']:.4f} m"
                    )
                else:
                    print("当前样本数量或几何秩不足，继续采集。")
                break

        quest_xyz_m = np.stack([np.asarray(sample["Q_T_H"], dtype=float)[:3, 3] for sample in samples])
        quest_centered = quest_xyz_m - quest_xyz_m.mean(axis=0)
        failures: list[str] = []
        if np.linalg.matrix_rank(quest_centered, tol=1.0e-6) < 3:
            failures.append("translation samples are rank deficient")
        else:
            W_T_Q = estimate_W_T_Q(target_xyz_m, quest_xyz_m)
            W_T_Q[:3, :3] = project_to_so3(W_T_Q[:3, :3])
        assert W_T_Q is not None
        errors = reprojection_error_stats(W_T_Q, target_xyz_m, quest_xyz_m)
        for sample in samples:
            sample["W_T_H"] = W_T_Q @ np.asarray(sample["Q_T_H"], dtype=float)
        rot_stats = rotation_quality(W_T_Q[:3, :3])

        if len(samples) < args_cli.min_point_count:
            failures.append(f"sample count {len(samples)} < {args_cli.min_point_count}")
        if errors["mean_m"] > args_cli.max_mean_error_m:
            failures.append(f"mean position error {errors['mean_m']:.4f} m > {args_cli.max_mean_error_m:.4f} m")
        if errors["max_m"] > args_cli.max_max_error_m:
            failures.append(f"max position error {errors['max_m']:.4f} m > {args_cli.max_max_error_m:.4f} m")
        if abs(rot_stats["det"] - 1.0) > 1.0e-6:
            failures.append(f"det(R) {rot_stats['det']:.9f} differs from 1.0")
        if rot_stats["orthogonality_error_fro"] > 1.0e-6:
            failures.append(f"orthogonality error {rot_stats['orthogonality_error_fro']:.9e} > 1e-6")

        print("=" * 72)
        print("平移标定结果")
        print(f"det(R) = {rot_stats['det']:.9f}")
        print(f"orthogonality_error_fro = {rot_stats['orthogonality_error_fro']:.9e}")
        print(f"anchor_position_error_m = {anchor_position_error_m:.9e}")
        print(f"mean_position_error_m = {errors['mean_m']:.6f}")
        print(f"max_position_error_m = {errors['max_m']:.6f}")

        rotation_direction_report: dict[str, Any] = {
            "enabled": True,
            "accepted": False,
            "skipped": True,
            "reason": "translation calibration failed",
            "R_rot_map": R_rot_map,
            "actions": [],
            "failures": [],
            "mean_axis_error_deg": None,
            "max_axis_error_deg": None,
            "thresholds": {
                "min_angle_deg": float(args_cli.rotation_min_angle_deg),
                "max_angle_deg": float(args_cli.rotation_max_angle_deg),
                "max_mean_axis_error_deg": float(args_cli.max_rotation_mean_axis_error_deg),
                "max_max_axis_error_deg": float(args_cli.max_rotation_max_axis_error_deg),
            },
        }
        if not failures:
            result = _capture_rotation_direction_calibration(
                xr_client,
                visualizers,
                scene,
                target_xyz_m,
                samples,
                W_T_Q,
                R_align,
                ref_Q_T_H,
                ref_tcp_T,
                sim,
                sim_dt,
                robot_default_joints,
            )
            if result is None:
                print("校准已中止。")
                return None
            R_rot_map = np.asarray(result["R_rot_map"], dtype=float)
            rotation_direction_report = result
            failures.extend([f"rotation direction: {failure}" for failure in result["failures"]])
        else:
            print("平移质量未通过，跳过旋转方向校准。")

        accepted = not failures
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = args_cli.output or os.path.join(args_cli.log_dir, f"xrobotoolkit_world_frame_calibration_{timestamp}.json")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        teleop_command = _teleop_command(output_path) if accepted else None
        recording_command = _recording_command(output_path) if accepted else None
        report = {
            "schema": "xrobotoolkit_world_frame_calibration_v1",
            "created_at": timestamp,
            "accepted": accepted,
            "failures": failures,
            "calibration_method": "isaaclab_viewport_openxr_seeded_relative_multipoint_with_rotation_direction_map",
            "pose_source": args_cli.pose_source,
            "anchor_key": args_cli.anchor_key,
            "anchor_threshold": float(args_cli.anchor_threshold),
            "capture_key": args_cli.capture_key,
            "retry_key": args_cli.retry_key,
            "exit_button": args_cli.exit_button,
            "openxr_to_ros_R": OPENXR_TO_ROS,
            "anchor_W_T_Q": anchor_W_T_Q,
            "W_T_Q": W_T_Q,
            "R_align": R_align,
            "R_align_rpy_rad": [0.0, 0.0, 0.0],
            "R_rot_map": R_rot_map,
            "rotation_direction_calibration": rotation_direction_report,
            "rotation_quality": rot_stats,
            "errors": errors,
            "quality_thresholds": {
                "min_point_count": int(args_cli.min_point_count),
                "max_mean_error_m": float(args_cli.max_mean_error_m),
                "max_max_error_m": float(args_cli.max_max_error_m),
                "relative_step_m": float(args_cli.relative_step_m),
                "relative_lift_m": float(args_cli.relative_lift_m),
                "anchor_position_error_m": anchor_position_error_m,
                "rotation_min_angle_deg": float(args_cli.rotation_min_angle_deg),
                "rotation_max_angle_deg": float(args_cli.rotation_max_angle_deg),
                "max_rotation_mean_axis_error_deg": float(args_cli.max_rotation_mean_axis_error_deg),
                "max_rotation_max_axis_error_deg": float(args_cli.max_rotation_max_axis_error_deg),
            },
            "calibration_targets": {
                "origin_world_xyz_m": target_config["origin_world_xyz_m"],
                "names": target_names,
                "descriptions": target_descriptions,
                "offset_xyz_m": target_offset_xyz_m,
                "world_xyz_m": target_xyz_m,
            },
            "anchor_sample": {
                "name": "anchor",
                "pose_xyzw": anchor_pose.copy(),
                "Q_T_H": ref_Q_T_H.copy(),
                "Qros_T_H": openxr_to_ros_tracking_T(ref_Q_T_H),
                "W_T_H": (W_T_Q @ ref_Q_T_H).copy(),
                "W_T_TCP_anchor": ref_tcp_T.copy(),
                "anchor_position_error_m": anchor_position_error_m,
            },
            "samples": samples,
            "teleop_command": teleop_command,
            "recording_command": recording_command,
        }
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=2, default=_json_default)

        print("=" * 72)
        print("最终校准结果")
        print(f"accepted = {accepted}")
        print(f"mean_position_error_m = {errors['mean_m']:.6f}")
        print(f"max_position_error_m = {errors['max_m']:.6f}")
        if rotation_direction_report.get("mean_axis_error_deg") is not None:
            print(f"mean_rotation_axis_error_deg = {rotation_direction_report['mean_axis_error_deg']:.3f}")
            print(f"max_rotation_axis_error_deg = {rotation_direction_report['max_axis_error_deg']:.3f}")
        print(f"校准报告已保存: {output_path}")
        if accepted:
            print("可直接运行的遥操作命令:")
            print(teleop_command)
            print("可直接运行的录制命令:")
            print(recording_command)
        else:
            print("校准未通过质量门槛，不输出可执行命令。")
            for failure in failures:
                print(f"  - {failure}")

        if accepted and args_cli.preview_after_fit:
            print(f"现在可移动手柄预览 TCP target frame；按 {args_cli.exit_button} 退出。")
            while simulation_app.is_running() and not _button_pressed(xr_client, args_cli.exit_button):
                draw(None, samples)
                time.sleep(1.0 / max(args_cli.rate_hz, 1.0))
        return output_path
    finally:
        try:
            xr_client.close()
            print("XR SDK 已关闭。")
        except Exception as exc:
            print(f"警告: XR SDK 关闭失败: {exc}")


def main() -> None:
    try:
        run_calibration()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
