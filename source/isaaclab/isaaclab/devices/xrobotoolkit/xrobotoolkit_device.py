# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""XRoboToolkit controller device for SE(3) teleoperation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ..device_base import DeviceBase, DeviceCfg


OPENXR_TO_ROBOT_BASE_AXIS_MAP = (
    (0.0, 0.0, -1.0),
    (-1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
)
"""Default axis map from OpenXR coordinates to Isaac/Piper robot-base coordinates."""


class XRoboToolkitDevice(DeviceBase):
    """XRoboToolkit controller device for Isaac Lab SE(3) + gripper teleoperation.

    The command layout is ``[dx, dy, dz, rx, ry, rz, gripper]``. Translation is in meters,
    rotation is a rotation vector in radians, and the gripper command is ``+1.0`` for open
    or ``-1.0`` for close. By default, OpenXR axes ``[right, up, back]`` are mapped into
    Isaac/Piper robot-base axes ``[forward, left, up]`` as ``[-z, -x, y]``.
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
        self._sim_device = cfg.sim_device
        self._delta_pos_axis_map = _axis_mapping_to_array(cfg.delta_pos_axis_map, "delta_pos_axis_map")
        self._delta_rot_axis_map = _axis_mapping_to_array(cfg.delta_rot_axis_map, "delta_rot_axis_map")
        self._xr_client = cfg.xr_client if cfg.xr_client is not None else self._create_xr_client()
        self._additional_callbacks: dict[str, Callable] = {}
        self._prev_active = False
        self._prev_reset_pressed = False
        self._ref_pos: np.ndarray | None = None
        self._ref_quat: np.ndarray | None = None

    def __str__(self) -> str:
        """Returns: A string containing the device information."""
        return (
            f"XRoboToolkit Controller for SE(3): {self.__class__.__name__}\n"
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

    def add_callback(self, key: Any, func: Callable):
        """Add a callback for an XRoboToolkit event key.

        Supported keys used by this device are ``START``, ``STOP``, and ``RESET``.
        """
        self._additional_callbacks[key] = func

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
            return self._command(np.zeros(3), np.zeros(3), 1.0)

        pose = self._read_pose(self.pose_source)
        if pose is None or self._ref_pos is None or self._ref_quat is None:
            return self._command(np.zeros(3), np.zeros(3), self._read_gripper())

        pos = pose[:3]
        quat = pose[3:7]
        delta_pos = (pos - self._ref_pos) * self.pos_sensitivity
        delta_quat = _quat_multiply(quat, _quat_conjugate(self._ref_quat))
        delta_rot = _quat_to_rotvec(delta_quat) * self.rot_sensitivity
        delta_pos = self._delta_pos_axis_map @ delta_pos
        delta_rot = self._delta_rot_axis_map @ delta_rot

        return self._command(delta_pos, delta_rot, self._read_gripper())

    def _create_xr_client(self):
        """Create the XRoboToolkit client lazily so importing Isaac Lab does not require the SDK."""
        try:
            from xrobotoolkit_teleop.common.xr_client import XrClient
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "XRoboToolkitDevice requires xrobotoolkit_teleop in the Isaac Lab Python environment. "
                "Install it with: ./isaaclab.sh -p -m pip install --no-deps -e "
                "/home/kongqingwei/XRoboToolkit-Teleop-Sample-Python"
            ) from exc

        return XrClient()

    def _capture_reference(self):
        pose = self._read_pose(self.pose_source)
        if pose is None:
            self._clear_reference()
            return
        self._ref_pos = pose[:3].copy()
        self._ref_quat = pose[3:7].copy()

    def _clear_reference(self):
        self._ref_pos = None
        self._ref_quat = None

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

    def _command(self, delta_pos: np.ndarray, delta_rot: np.ndarray, gripper: float) -> torch.Tensor:
        command = np.concatenate([delta_pos, delta_rot])
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
    delta_pos_axis_map: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] = (
        OPENXR_TO_ROBOT_BASE_AXIS_MAP
    )
    delta_rot_axis_map: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] = (
        OPENXR_TO_ROBOT_BASE_AXIS_MAP
    )
    retargeters: None = None
    xr_client: Any | None = None
    class_type: type[DeviceBase] = XRoboToolkitDevice


def _axis_mapping_to_array(mapping: Any, name: str) -> np.ndarray:
    array = np.asarray(mapping, dtype=float)
    if array.shape != (3, 3) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite 3x3 matrix, got shape {array.shape}")
    return array


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
