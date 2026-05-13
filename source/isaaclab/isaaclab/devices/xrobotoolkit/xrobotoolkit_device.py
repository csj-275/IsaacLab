# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""XRoboToolkit controller device for SE(3) teleoperation."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch

from ..device_base import DeviceBase, DeviceCfg


XR_SDK_TO_ROS_BASE_POS_AXIS_MAP = (
    (0.0, 0.0, -1.0),
    (-1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
)
"""Default position map from XR SDK input coordinates to ROS base axes: X forward, Y left, Z up."""

XR_SDK_TO_ROS_BASE_ROT_AXIS_MAP = (
    (0.0, 0.0, -1.0),
    (-1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
)
"""Default rotation map calibrated from XR SDK input rotation vectors to robot end-effector effects."""


class XRoboToolkitDevice(DeviceBase):
    """XRoboToolkit controller device for Isaac Lab SE(3) + gripper teleoperation.

    In relative mode, the command layout is ``[dx, dy, dz, rx, ry, rz, gripper]``.
    In absolute mode, the command layout is ``[x, y, z, qw, qx, qy, qz, gripper]``.
    By default, mapping mode is ``world_frame_calibrated``. When a calibration JSON
    is provided, absolute control maps full controller poses by ``W_T_Q`` and
    applies split world-frame translation plus independently mapped SO(3) rotation.
    Relative control keeps the 6D delta-vector interface and uses ``W_T_Q[:3,:3]``
    with ``R_rot_map``. Without a calibration JSON, the device falls back to the
    built-in OpenXR-to-ROS axis maps.
    """

    def __init__(self, cfg: XRoboToolkitDeviceCfg):
        """Initialize the XRoboToolkit teleoperation device.

        Args:
            cfg: Configuration object for XRoboToolkit controller input.
        """
        super().__init__(retargeters=None)
        self.pose_source = cfg.pose_source
        self.control_trigger = cfg.control_trigger
        self.gripper_trigger = cfg.gripper_trigger
        self.reset_button = cfg.reset_button
        self.pos_sensitivity = cfg.pos_sensitivity
        self.rot_sensitivity = cfg.rot_sensitivity
        self.activation_threshold = cfg.activation_threshold
        self.gripper_threshold = cfg.gripper_threshold
        self.gripper_term = cfg.gripper_term
        self.control_mode = cfg.control_mode
        if self.control_mode not in ("relative", "absolute"):
            raise ValueError(f"Unsupported XRoboToolkit control mode: {self.control_mode}")
        self.mapping_mode = cfg.mapping_mode
        if self.mapping_mode not in ("world_frame_calibrated", "axis_map"):
            raise ValueError(f"Unsupported XRoboToolkit mapping mode: {self.mapping_mode}")
        self.debug_mapping = cfg.debug_mapping
        self.debug_mapping_interval = cfg.debug_mapping_interval
        self._sim_device = cfg.sim_device
        self._delta_pos_axis_map = _axis_mapping_to_array(cfg.delta_pos_axis_map, "delta_pos_axis_map")
        self._delta_rot_axis_map = _axis_mapping_to_array(cfg.delta_rot_axis_map, "delta_rot_axis_map")
        self._calibrated_axis_map: np.ndarray | None = None
        self._calibrated_rot_map: np.ndarray | None = None
        self._calibrated_world_from_xr: np.ndarray | None = None
        if self.mapping_mode == "world_frame_calibrated" and cfg.calibration_json is not None:
            calib = _load_calibration(cfg.calibration_json)
            self._calibrated_axis_map = calib["axis_map"]
            self._calibrated_rot_map = calib["rot_map"]
            self._calibrated_world_from_xr = calib["world_from_xr"]
        elif self.mapping_mode == "world_frame_calibrated":
            print(
                "XRoboToolkitDevice: mapping_mode=world_frame_calibrated is using uncalibrated fallback "
                "(no calibration_json provided); provide --xrobotoolkit_calibration_json to use W_T_Q + R_rot_map.",
                flush=True,
            )
        self._ee_pose_provider = cfg.ee_pose_provider
        self._xr_client = cfg.xr_client if cfg.xr_client is not None else self._create_xr_client()
        self._additional_callbacks: dict[str, Callable] = {}
        self._prev_active = False
        self._prev_reset_pressed = False
        self._ref_pos: np.ndarray | None = None
        self._ref_quat: np.ndarray | None = None
        self._ref_robot_pos: np.ndarray | None = None
        self._ref_robot_quat: np.ndarray | None = None
        self._ref_controller_T: np.ndarray | None = None
        self._ref_robot_T: np.ndarray | None = None
        self._last_debug_mapping_time = -float("inf")

    def __str__(self) -> str:
        """Returns: A string containing the device information."""
        return (
            f"XRoboToolkit Controller for SE(3): {self.__class__.__name__}\n"
            f"\tControl mode: {self.control_mode}\n"
            f"\tMapping mode: {self.mapping_mode}\n"
            f"\tCalibrated: {self._calibrated_axis_map is not None}\n"
            f"\tPose source: {self.pose_source}\n"
            f"\tControl trigger: {self.control_trigger}\n"
            f"\tGripper trigger: {self.gripper_trigger}\n"
            f"\tReset button: {self.reset_button}"
        )

    def reset(self):
        """Reset the reference pose and edge-trigger state."""
        self._prev_active = False
        self._prev_reset_pressed = False
        self._ref_pos = None
        self._ref_quat = None
        self._ref_robot_pos = None
        self._ref_robot_quat = None
        self._ref_controller_T = None
        self._ref_robot_T = None

    def add_callback(self, key: Any, func: Callable):
        """Add a callback for an XRoboToolkit event key.

        Supported keys used by this device are ``START``, ``STOP``, and ``RESET``.
        """
        self._additional_callbacks[key] = func

    def set_absolute_pose_provider(self, provider: Callable[[], tuple[Any, Any]]):
        """Set the provider used to anchor robot end-effector pose in absolute mode."""
        self._ee_pose_provider = provider

    def advance(self) -> torch.Tensor:
        """Read XRoboToolkit input and return an Isaac Lab SE(3) + gripper command."""
        reset_pressed = self._read_button(self.reset_button)
        if reset_pressed and not self._prev_reset_pressed:
            self._clear_reference()
            self._run_callback("RESET")

        active = self._read_trigger(self.control_trigger, self.activation_threshold)
        if active and not self._prev_active:
            self._capture_reference()
            self._run_callback("START")
        elif not active and self._prev_active:
            self._clear_reference()
            self._run_callback("STOP")

        self._prev_reset_pressed = reset_pressed
        self._prev_active = active

        if not active:
            return self._inactive_command()

        pose = self._read_pose(self.pose_source)
        if pose is None or self._ref_pos is None or self._ref_quat is None:
            return self._inactive_command(self._read_gripper())

        pos = pose[:3]
        quat = pose[3:7]
        raw_delta_pos = (pos - self._ref_pos) * self.pos_sensitivity
        delta_quat = _quat_multiply(quat, _quat_conjugate(self._ref_quat))
        raw_delta_rot = _quat_to_rotvec(delta_quat) * self.rot_sensitivity
        if self.mapping_mode == "world_frame_calibrated" and self._calibrated_axis_map is not None:
            assert self._calibrated_rot_map is not None
            delta_pos = self._calibrated_axis_map @ raw_delta_pos
            delta_rot_remapped = self._calibrated_axis_map @ raw_delta_rot
            delta_rot = self._calibrated_rot_map @ delta_rot_remapped
        else:
            delta_pos = self._delta_pos_axis_map @ raw_delta_pos
            delta_rot = self._delta_rot_axis_map @ raw_delta_rot

        if self.control_mode == "absolute":
            if self._ref_robot_pos is None or self._ref_robot_quat is None:
                return self._inactive_command(self._read_gripper())
            if (
                self.mapping_mode == "world_frame_calibrated"
                and self._calibrated_world_from_xr is not None
                and self._calibrated_rot_map is not None
            ):
                target_pos, target_quat, delta_pos, delta_rot = self._compute_world_frame_calibrated_target(pos, quat)
                self._debug_mapping(raw_delta_pos, delta_pos, raw_delta_rot, delta_rot)
                return self._absolute_command(target_pos, target_quat, self._read_gripper())

            self._debug_mapping(raw_delta_pos, delta_pos, raw_delta_rot, delta_rot)
            target_pos = self._ref_robot_pos + delta_pos
            target_quat = _xyzw_to_wxyz(
                _quat_multiply(_rotvec_to_quat(delta_rot), _wxyz_to_xyzw(self._ref_robot_quat))
            )
            return self._absolute_command(target_pos, target_quat, self._read_gripper())
        else:
            self._debug_mapping(raw_delta_pos, delta_pos, raw_delta_rot, delta_rot)
            return self._relative_command(delta_pos, delta_rot, self._read_gripper())

    @property
    def command_dim(self) -> int:
        """Dimension of the command emitted by this device."""
        base_dim = 7 if self.control_mode == "absolute" else 6
        return base_dim + int(self.gripper_term)

    def _create_xr_client(self):
        """Create the XRoboToolkit client lazily so importing Isaac Lab does not require the SDK."""
        try:
            from xrobotoolkit_teleop.common.xr_client import XrClient
        except ModuleNotFoundError as exc:
            setup_hint = "bash /workspace/isaaclab/scripts/tools/setup_xrobotoolkit_env.sh"
            if exc.name == "xrobotoolkit_teleop":
                raise ModuleNotFoundError(
                    "XRoboToolkitDevice requires xrobotoolkit_teleop in the Isaac Lab Python environment. "
                    f"In the Docker container, install it with: {setup_hint}"
                ) from exc
            if exc.name == "xrobotoolkit_sdk":
                raise ModuleNotFoundError(
                    "XRoboToolkitDevice requires xrobotoolkit_sdk for XRoboToolkit PC Service access. "
                    f"In the Docker container, build and install it with: {setup_hint}"
                ) from exc
            raise

        return XrClient()

    def _capture_reference(self):
        pose = self._read_pose(self.pose_source)
        if pose is None:
            self._clear_reference()
            return
        self._ref_pos = pose[:3].copy()
        self._ref_quat = pose[3:7].copy()
        self._ref_controller_T = _make_transform(_quat_to_matrix(self._ref_quat), self._ref_pos)
        if self.control_mode == "absolute":
            self._ref_robot_pos, self._ref_robot_quat = self._read_robot_pose()
            self._ref_robot_T = _make_transform(
                _quat_to_matrix(_wxyz_to_xyzw(self._ref_robot_quat)),
                self._ref_robot_pos,
            )

    def _clear_reference(self):
        self._ref_pos = None
        self._ref_quat = None
        self._ref_robot_pos = None
        self._ref_robot_quat = None
        self._ref_controller_T = None
        self._ref_robot_T = None

    def _compute_world_frame_calibrated_target(
        self, pos: np.ndarray, quat: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self._calibrated_world_from_xr is None or self._calibrated_rot_map is None:
            raise RuntimeError("world_frame_calibrated target requires loaded W_T_Q and R_rot_map.")
        if self._ref_controller_T is None or self._ref_robot_T is None:
            raise RuntimeError("world_frame_calibrated target requires captured controller and TCP references.")

        controller_T = _make_transform(_quat_to_matrix(quat), pos)
        W_T_H0 = self._calibrated_world_from_xr @ self._ref_controller_T
        W_T_Ht = self._calibrated_world_from_xr @ controller_T

        delta_pos = self.pos_sensitivity * (W_T_Ht[:3, 3] - W_T_H0[:3, 3])
        R_delta_controller = _project_to_so3(W_T_Ht[:3, :3] @ W_T_H0[:3, :3].T)
        rotvec_controller = _so3_log(R_delta_controller)
        delta_rot = self.rot_sensitivity * (self._calibrated_rot_map @ rotvec_controller)

        target_pos = self._ref_robot_T[:3, 3] + delta_pos
        target_rot = _project_to_so3(_so3_exp(delta_rot) @ self._ref_robot_T[:3, :3])
        target_quat = _xyzw_to_wxyz(_matrix_to_quat(target_rot))
        return target_pos, target_quat, delta_pos, delta_rot

    def _read_robot_pose(self) -> tuple[np.ndarray, np.ndarray]:
        if self._ee_pose_provider is None:
            raise RuntimeError("XRoboToolkit absolute mode requires an end-effector pose provider.")

        pos, quat = self._ee_pose_provider()
        pos = np.asarray(pos, dtype=float)
        quat = _normalize_quat(np.asarray(quat, dtype=float))
        if pos.shape != (3,) or quat is None or not np.all(np.isfinite(pos)):
            raise RuntimeError("XRoboToolkit absolute mode received an invalid end-effector reference pose.")
        return pos.copy(), quat.copy()

    def _read_pose(self, name: str) -> np.ndarray | None:
        pose = self._xr_client.get_pose_by_name(name)
        if pose is None:
            return None

        pose = np.asarray(pose, dtype=float)
        if pose.shape[0] < 7:
            return None
        pose = pose[:7]
        quat = _normalize_quat(pose[3:7])
        if quat is None:
            return None
        pose[3:7] = quat
        if not np.all(np.isfinite(pose[:3])):
            return None
        return pose

    def _read_trigger(self, name: str, threshold: float) -> bool:
        return float(self._xr_client.get_key_value_by_name(name)) >= threshold

    def _read_button(self, name: str) -> bool:
        return bool(self._xr_client.get_button_state_by_name(name))

    def _read_gripper(self) -> float:
        close_gripper = self._read_trigger(self.gripper_trigger, self.gripper_threshold)
        return -1.0 if close_gripper else 1.0

    def _run_callback(self, key: str):
        callback = self._additional_callbacks.get(key)
        if callback is not None:
            callback()

    def _debug_mapping(
        self,
        raw_delta_pos: np.ndarray,
        mapped_delta_pos: np.ndarray,
        raw_delta_rot: np.ndarray,
        mapped_delta_rot: np.ndarray,
    ):
        if not self.debug_mapping:
            return

        now = time.monotonic()
        if self.debug_mapping_interval > 0.0 and now - self._last_debug_mapping_time < self.debug_mapping_interval:
            return
        self._last_debug_mapping_time = now

        print(
            "[XRoboToolkit mapping] "
            f"mode={self.control_mode} "
            f"mapping_mode={self.mapping_mode} "
            f"calibrated={str(self._calibrated_axis_map is not None).lower()} "
            f"raw_pos={_format_vector(raw_delta_pos)} "
            f"mapped_pos={_format_vector(mapped_delta_pos)} "
            f"raw_rot={_format_vector(raw_delta_rot)} "
            f"mapped_rot={_format_vector(mapped_delta_rot)}",
            flush=True,
        )

    def _inactive_command(self, gripper: float = 1.0) -> torch.Tensor:
        if self.control_mode == "absolute":
            target_pos = self._ref_robot_pos if self._ref_robot_pos is not None else np.zeros(3)
            target_quat = self._ref_robot_quat if self._ref_robot_quat is not None else np.array([1.0, 0.0, 0.0, 0.0])
            return self._absolute_command(target_pos, target_quat, gripper)
        else:
            return self._relative_command(np.zeros(3), np.zeros(3), gripper)

    def _relative_command(self, delta_pos: np.ndarray, delta_rot: np.ndarray, gripper: float) -> torch.Tensor:
        command = np.concatenate([delta_pos, delta_rot])
        if self.gripper_term:
            command = np.append(command, gripper)
        return torch.tensor(command, dtype=torch.float32, device=self._sim_device)

    def _absolute_command(self, target_pos: np.ndarray, target_quat: np.ndarray, gripper: float) -> torch.Tensor:
        command = np.concatenate([target_pos, target_quat])
        if self.gripper_term:
            command = np.append(command, gripper)
        return torch.tensor(command, dtype=torch.float32, device=self._sim_device)


@dataclass
class XRoboToolkitDeviceCfg(DeviceCfg):
    """Configuration for XRoboToolkit SE(3) teleoperation."""

    pose_source: str = "right_controller"
    control_trigger: str = "right_grip"
    gripper_trigger: str = "right_trigger"
    reset_button: str = "A"
    pos_sensitivity: float = 1.0
    rot_sensitivity: float = 1.0
    activation_threshold: float = 0.5
    gripper_threshold: float = 0.5
    gripper_term: bool = True
    control_mode: Literal["relative", "absolute"] = "absolute"
    mapping_mode: Literal["world_frame_calibrated", "axis_map"] = "world_frame_calibrated"
    """Coordinate mapping mode for XR controller deltas."""
    debug_mapping: bool = False
    debug_mapping_interval: float = 0.5
    delta_pos_axis_map: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] = (
        XR_SDK_TO_ROS_BASE_POS_AXIS_MAP
    )
    delta_rot_axis_map: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] = (
        XR_SDK_TO_ROS_BASE_ROT_AXIS_MAP
    )
    calibration_json: str | None = None
    """Path to a piper world-frame calibration JSON.

    Used only in mapping_mode="world_frame_calibrated". In absolute mode, W_T_Q
    maps captured controller poses into the robot world frame before split
    translation/rotation target synthesis. In relative mode, W_T_Q[:3,:3]
    replaces delta_pos_axis_map and delta_rot_axis_map, and R_rot_map is applied
    to rotation deltas on top of the axis mapping. When omitted, the device uses
    the built-in axis-map fallback.
    """
    retargeters: None = None
    ee_pose_provider: Callable[[], tuple[Any, Any]] | None = None
    xr_client: Any | None = None
    class_type: type[DeviceBase] = XRoboToolkitDevice


def _load_calibration(json_path: str) -> dict[str, np.ndarray]:
    """Load a piper world-frame calibration JSON and return transform/axis/rot maps.

    Args:
        json_path: Path to the calibration JSON file.

    Returns:
        Dict with keys "world_from_xr" (4x4), "axis_map" (3x3), and "rot_map" (3x3).

    Raises:
        FileNotFoundError: If the JSON file does not exist.
        ValueError: If the calibration is not accepted or the data is invalid.
    """
    from xrobotoolkit_teleop.hardware.piper_world_frame_mapping import (
        load_piper_world_calibration_json,
        project_to_so3,
    )

    data = load_piper_world_calibration_json(json_path)
    world_from_xr = _validate_transform(np.asarray(data["W_T_Q"], dtype=float), "W_T_Q")
    axis_map = project_to_so3(world_from_xr[:3, :3])
    rot_map = project_to_so3(data["R_rot_map"])
    return {"world_from_xr": world_from_xr, "axis_map": axis_map, "rot_map": rot_map}


def _axis_mapping_to_array(mapping: Any, name: str) -> np.ndarray:
    array = np.asarray(mapping, dtype=float)
    if array.shape != (3, 3) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite 3x3 matrix, got shape {array.shape}")
    return array


def _format_vector(vector: np.ndarray) -> str:
    return np.array2string(np.asarray(vector, dtype=float), precision=4, suppress_small=True)


def _make_transform(rot: np.ndarray | None = None, pos: np.ndarray | None = None) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    if rot is not None:
        transform[:3, :3] = _project_to_so3(rot)
    if pos is not None:
        pos = np.asarray(pos, dtype=float)
        if pos.shape != (3,) or not np.all(np.isfinite(pos)):
            raise ValueError(f"Transform position must be a finite 3-vector, got shape {pos.shape}")
        transform[:3, 3] = pos
    return transform


def _validate_transform(transform: np.ndarray, name: str) -> np.ndarray:
    transform = np.asarray(transform, dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f"{name} must be a finite 4x4 matrix, got shape {transform.shape}")
    if not np.allclose(transform[3, :], np.array([0.0, 0.0, 0.0, 1.0]), atol=1.0e-9):
        raise ValueError(f"{name} bottom row must be [0, 0, 0, 1]")
    checked = transform.copy()
    checked[:3, :3] = _project_to_so3(checked[:3, :3])
    return checked


def _project_to_so3(rot: np.ndarray) -> np.ndarray:
    rot = np.asarray(rot, dtype=float)
    if rot.shape != (3, 3) or not np.all(np.isfinite(rot)):
        raise ValueError(f"Rotation must be a finite 3x3 matrix, got shape {rot.shape}")
    u, _, vt = np.linalg.svd(rot)
    projected = u @ vt
    if np.linalg.det(projected) < 0.0:
        u[:, -1] *= -1.0
        projected = u @ vt
    return projected


def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = _normalize_quat(quat)
    if quat is None:
        return np.eye(3, dtype=float)
    x, y, z, w = quat
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _matrix_to_quat(rot: np.ndarray) -> np.ndarray:
    rot = _project_to_so3(rot)
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        quat = np.array(
            [
                (rot[2, 1] - rot[1, 2]) / s,
                (rot[0, 2] - rot[2, 0]) / s,
                (rot[1, 0] - rot[0, 1]) / s,
                0.25 * s,
            ],
            dtype=float,
        )
    else:
        axis = int(np.argmax(np.diag(rot)))
        if axis == 0:
            s = 2.0 * np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2])
            quat = np.array(
                [
                    0.25 * s,
                    (rot[0, 1] + rot[1, 0]) / s,
                    (rot[0, 2] + rot[2, 0]) / s,
                    (rot[2, 1] - rot[1, 2]) / s,
                ],
                dtype=float,
            )
        elif axis == 1:
            s = 2.0 * np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2])
            quat = np.array(
                [
                    (rot[0, 1] + rot[1, 0]) / s,
                    0.25 * s,
                    (rot[1, 2] + rot[2, 1]) / s,
                    (rot[0, 2] - rot[2, 0]) / s,
                ],
                dtype=float,
            )
        else:
            s = 2.0 * np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1])
            quat = np.array(
                [
                    (rot[0, 2] + rot[2, 0]) / s,
                    (rot[1, 2] + rot[2, 1]) / s,
                    0.25 * s,
                    (rot[1, 0] - rot[0, 1]) / s,
                ],
                dtype=float,
            )
    quat = _normalize_quat(quat)
    if quat is None:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    if quat[3] < 0.0:
        quat = -quat
    return quat


def _normalize_quat(quat: np.ndarray) -> np.ndarray | None:
    quat = np.asarray(quat, dtype=float)
    if quat.shape != (4,) or not np.all(np.isfinite(quat)):
        return None
    norm = np.linalg.norm(quat)
    if norm < 1.0e-12:
        return None
    return quat / norm


def _quat_conjugate(quat: np.ndarray) -> np.ndarray:
    return np.array([-quat[0], -quat[1], -quat[2], quat[3]], dtype=float)


def _quat_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = lhs
    x2, y2, z2, w2 = rhs
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=float,
    )


def _quat_to_rotvec(quat: np.ndarray) -> np.ndarray:
    quat = _normalize_quat(quat)
    if quat is None:
        return np.zeros(3)
    if quat[3] < 0.0:
        quat = -quat

    xyz = quat[:3]
    xyz_norm = np.linalg.norm(xyz)
    if xyz_norm < 1.0e-12:
        return 2.0 * xyz

    angle = 2.0 * np.arctan2(xyz_norm, quat[3])
    return xyz / xyz_norm * angle


def _rotvec_to_quat(rotvec: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(rotvec)
    if angle < 1.0e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    axis = rotvec / angle
    return np.concatenate([axis * np.sin(angle / 2.0), [np.cos(angle / 2.0)]])


def _so3_log(rot: np.ndarray) -> np.ndarray:
    rot = _project_to_so3(rot)
    cos_theta = float(np.clip((np.trace(rot) - 1.0) * 0.5, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))
    if theta < 1.0e-12:
        return np.zeros(3, dtype=float)

    if np.pi - theta < 1.0e-5:
        axis = np.array(
            [
                np.sqrt(max((rot[0, 0] + 1.0) * 0.5, 0.0)),
                np.sqrt(max((rot[1, 1] + 1.0) * 0.5, 0.0)),
                np.sqrt(max((rot[2, 2] + 1.0) * 0.5, 0.0)),
            ],
            dtype=float,
        )
        axis[1] = np.copysign(axis[1], rot[0, 1] + rot[1, 0])
        axis[2] = np.copysign(axis[2], rot[0, 2] + rot[2, 0])
        norm = float(np.linalg.norm(axis))
        if norm < 1.0e-9:
            axis = np.array([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]])
            norm = float(np.linalg.norm(axis))
        if norm < 1.0e-9:
            return np.zeros(3, dtype=float)
        return theta * axis / norm

    axis = np.array([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]], dtype=float)
    axis /= 2.0 * np.sin(theta)
    return theta * axis


def _so3_exp(rotvec: np.ndarray) -> np.ndarray:
    return _quat_to_matrix(_rotvec_to_quat(np.asarray(rotvec, dtype=float)))


def _wxyz_to_xyzw(quat: np.ndarray) -> np.ndarray:
    return np.array([quat[1], quat[2], quat[3], quat[0]], dtype=float)


def _xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = _normalize_quat(quat)
    if quat is None:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return np.array([quat[3], quat[0], quat[1], quat[2]], dtype=float)
