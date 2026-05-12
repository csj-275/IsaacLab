# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the XRoboToolkit teleoperation device."""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app so isaaclab.devices can import OpenXR/pxr side modules
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import numpy as np
import torch

from isaaclab.devices.xrobotoolkit.xrobotoolkit_device import XRoboToolkitDevice, XRoboToolkitDeviceCfg


IDENTITY_AXIS_MAP = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


class FakeXrClient:
    """Minimal XR client stub for deterministic device tests."""

    def __init__(self):
        self.pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        self.keys = {"right_grip": 0.0, "right_trigger": 0.0}
        self.buttons = {"A": False}

    def get_pose_by_name(self, name: str):
        assert name == "right_controller"
        return self.pose.copy()

    def get_key_value_by_name(self, name: str) -> float:
        return self.keys[name]

    def get_button_state_by_name(self, name: str) -> bool:
        return self.buttons[name]


def _pose(pos, rotvec=(0.0, 0.0, 0.0)) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=float)
    angle = np.linalg.norm(rotvec)
    if angle < 1.0e-12:
        quat = np.array([0.0, 0.0, 0.0, 1.0])
    else:
        axis = rotvec / angle
        quat = np.concatenate([axis * np.sin(angle / 2.0), [np.cos(angle / 2.0)]])
    return np.concatenate([np.asarray(pos, dtype=float), quat])


def test_xrobotoolkit_device_cfg_defaults_to_absolute_control_mode():
    client = FakeXrClient()
    cfg = XRoboToolkitDeviceCfg(xr_client=client)
    device = XRoboToolkitDevice(cfg)

    assert cfg.control_mode == "absolute"
    assert cfg.mapping_mode == "world_frame_calibrated"
    assert cfg.debug_mapping is False
    assert device.command_dim == 8
    action = device.advance()
    assert action.shape == (8,)
    np.testing.assert_allclose(action.cpu().numpy(), np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]))


def test_xrobotoolkit_device_control_edges_and_action_mapping():
    client = FakeXrClient()
    device = XRoboToolkitDevice(XRoboToolkitDeviceCfg(control_mode="relative", xr_client=client))
    events = []
    device.add_callback("START", lambda: events.append("START"))
    device.add_callback("STOP", lambda: events.append("STOP"))
    device.add_callback("RESET", lambda: events.append("RESET"))

    action = device.advance()
    assert isinstance(action, torch.Tensor)
    assert action.shape == (7,)
    np.testing.assert_allclose(action.cpu().numpy(), np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))

    client.keys["right_grip"] = 1.0
    action = device.advance()
    assert events == ["START"]
    np.testing.assert_allclose(action.cpu().numpy(), np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))

    client.pose = _pose([0.0, 0.0, -0.1], [0.0, 0.0, -np.pi / 2.0])
    client.keys["right_trigger"] = 1.0
    action = device.advance()
    np.testing.assert_allclose(
        action.cpu().numpy(),
        np.array([0.1, 0.0, 0.0, np.pi / 2.0, 0.0, 0.0, -1.0]),
        atol=1.0e-6,
    )

    client.buttons["A"] = True
    _ = device.advance()
    assert events == ["START", "RESET"]

    client.buttons["A"] = False
    client.keys["right_grip"] = 0.0
    action = device.advance()
    assert events == ["START", "RESET", "STOP"]
    np.testing.assert_allclose(action.cpu().numpy(), np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))


def test_xrobotoolkit_device_absolute_control_mode_anchors_robot_pose():
    client = FakeXrClient()
    robot_ref_pos = np.array([0.4, 0.1, 0.2])
    robot_ref_quat = np.array([1.0, 0.0, 0.0, 0.0])
    device = XRoboToolkitDevice(
        XRoboToolkitDeviceCfg(
            xr_client=client,
            ee_pose_provider=lambda: (robot_ref_pos.copy(), robot_ref_quat.copy()),
        )
    )
    events = []
    device.add_callback("START", lambda: events.append("START"))

    action = device.advance()
    assert action.shape == (8,)
    np.testing.assert_allclose(action.cpu().numpy(), np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]))

    client.keys["right_grip"] = 1.0
    action = device.advance()
    assert events == ["START"]
    np.testing.assert_allclose(action.cpu().numpy(), np.array([0.4, 0.1, 0.2, 1.0, 0.0, 0.0, 0.0, 1.0]))

    robot_ref_pos[:] = np.array([9.0, 9.0, 9.0])
    client.pose = _pose([0.0, 0.0, -0.1], [0.0, 0.0, -np.pi / 2.0])
    client.keys["right_trigger"] = 1.0
    action = device.advance()
    np.testing.assert_allclose(
        action.cpu().numpy(),
        np.array([0.5, 0.1, 0.2, np.cos(np.pi / 4.0), np.sin(np.pi / 4.0), 0.0, 0.0, -1.0]),
        atol=1.0e-6,
    )


def test_xrobotoolkit_device_default_ros_base_position_mapping():
    client = FakeXrClient()
    device = XRoboToolkitDevice(XRoboToolkitDeviceCfg(control_mode="relative", xr_client=client))

    client.keys["right_grip"] = 1.0
    _ = device.advance()

    client.pose = _pose([0.0, 0.0, -0.1])
    action = device.advance()
    np.testing.assert_allclose(action.cpu().numpy()[:3], np.array([0.1, 0.0, 0.0]), atol=1.0e-6)

    client.pose = _pose([0.0, 0.0, 0.1])
    action = device.advance()
    np.testing.assert_allclose(action.cpu().numpy()[:3], np.array([-0.1, 0.0, 0.0]), atol=1.0e-6)

    client.pose = _pose([-0.1, 0.0, 0.0])
    action = device.advance()
    np.testing.assert_allclose(action.cpu().numpy()[:3], np.array([0.0, 0.1, 0.0]), atol=1.0e-6)

    client.pose = _pose([0.1, 0.0, 0.0])
    action = device.advance()
    np.testing.assert_allclose(action.cpu().numpy()[:3], np.array([0.0, -0.1, 0.0]), atol=1.0e-6)

    client.pose = _pose([0.0, 0.1, 0.0])
    action = device.advance()
    np.testing.assert_allclose(action.cpu().numpy()[:3], np.array([0.0, 0.0, 0.1]), atol=1.0e-6)

    client.pose = _pose([0.0, -0.1, 0.0])
    action = device.advance()
    np.testing.assert_allclose(action.cpu().numpy()[:3], np.array([0.0, 0.0, -0.1]), atol=1.0e-6)


def test_xrobotoolkit_device_default_world_frame_calibrated_falls_back_to_axis_map(capsys):
    client = FakeXrClient()
    device = XRoboToolkitDevice(XRoboToolkitDeviceCfg(control_mode="relative", xr_client=client))

    output = capsys.readouterr().out
    assert "mapping_mode=world_frame_calibrated is using uncalibrated fallback" in output
    assert device.mapping_mode == "world_frame_calibrated"

    client.keys["right_grip"] = 1.0
    _ = device.advance()

    client.pose = _pose([0.0, 0.0, -0.1], [0.0, 0.0, -0.2])
    action = device.advance()
    np.testing.assert_allclose(
        action.cpu().numpy()[:6],
        np.array([0.1, 0.0, 0.0, 0.2, 0.0, 0.0]),
        atol=1.0e-6,
    )


def test_xrobotoolkit_device_default_calibrated_rotation_mapping():
    client = FakeXrClient()
    device = XRoboToolkitDevice(XRoboToolkitDeviceCfg(control_mode="relative", xr_client=client))

    client.keys["right_grip"] = 1.0
    _ = device.advance()

    client.pose = _pose([0.0, 0.0, 0.0], [0.0, 0.0, -0.2])
    action = device.advance()
    np.testing.assert_allclose(action.cpu().numpy()[3:6], np.array([0.2, 0.0, 0.0]), atol=1.0e-6)

    client.pose = _pose([0.0, 0.0, 0.0], [0.0, 0.0, 0.2])
    action = device.advance()
    np.testing.assert_allclose(action.cpu().numpy()[3:6], np.array([-0.2, 0.0, 0.0]), atol=1.0e-6)

    client.pose = _pose([0.0, 0.0, 0.0], [-0.2, 0.0, 0.0])
    action = device.advance()
    np.testing.assert_allclose(action.cpu().numpy()[3:6], np.array([0.0, 0.2, 0.0]), atol=1.0e-6)

    client.pose = _pose([0.0, 0.0, 0.0], [0.2, 0.0, 0.0])
    action = device.advance()
    np.testing.assert_allclose(action.cpu().numpy()[3:6], np.array([0.0, -0.2, 0.0]), atol=1.0e-6)

    client.pose = _pose([0.0, 0.0, 0.0], [0.0, 0.2, 0.0])
    action = device.advance()
    np.testing.assert_allclose(action.cpu().numpy()[3:6], np.array([0.0, 0.0, 0.2]), atol=1.0e-6)

    client.pose = _pose([0.0, 0.0, 0.0], [0.0, -0.2, 0.0])
    action = device.advance()
    np.testing.assert_allclose(action.cpu().numpy()[3:6], np.array([0.0, 0.0, -0.2]), atol=1.0e-6)


def test_xrobotoolkit_device_axis_mapping_can_be_overridden():
    client = FakeXrClient()
    device = XRoboToolkitDevice(
        XRoboToolkitDeviceCfg(
            control_mode="relative",
            xr_client=client,
            delta_pos_axis_map=IDENTITY_AXIS_MAP,
            delta_rot_axis_map=IDENTITY_AXIS_MAP,
        )
    )

    client.keys["right_grip"] = 1.0
    _ = device.advance()

    client.pose = _pose([0.1, 0.2, -0.3], [0.1, 0.2, -0.3])
    action = device.advance()
    np.testing.assert_allclose(action.cpu().numpy()[:6], np.array([0.1, 0.2, -0.3, 0.1, 0.2, -0.3]), atol=1.0e-6)


def test_xrobotoolkit_device_debug_mapping_does_not_change_action(capsys):
    client = FakeXrClient()
    device = XRoboToolkitDevice(
        XRoboToolkitDeviceCfg(
            control_mode="relative",
            debug_mapping=True,
            debug_mapping_interval=0.0,
            xr_client=client,
        )
    )

    client.keys["right_grip"] = 1.0
    _ = device.advance()
    capsys.readouterr()

    client.pose = _pose([0.1, 0.2, -0.3], [0.1, 0.2, -0.3])
    action = device.advance()

    np.testing.assert_allclose(
        action.cpu().numpy()[:6],
        np.array([0.3, -0.1, 0.2, 0.3, -0.1, 0.2]),
        atol=1.0e-6,
    )
    output = capsys.readouterr().out
    assert "[XRoboToolkit mapping]" in output
    assert "mode=relative" in output
    assert "mapping_mode=world_frame_calibrated" in output
    assert "calibrated=false" in output
    assert "raw_pos=" in output
    assert "mapped_pos=" in output
    assert "raw_rot=" in output
    assert "mapped_rot=" in output


def _make_calib_json(W_T_Q, R_rot_map=None):
    """Build a minimal accepted calibration JSON dict."""
    if R_rot_map is None:
        R_rot_map = np.eye(3, dtype=float)
    return {
        "accepted": True,
        "W_T_Q": W_T_Q.tolist(),
        "R_align": np.eye(3, dtype=float).tolist(),
        "R_rot_map": R_rot_map.tolist(),
        "R_align_rpy_rad": [0.0, 0.0, 0.0],
    }


def test_calibration_json_identity(tmp_path):
    """Calibration with identity W_T_Q and identity R_rot_map matches identity axis maps."""
    import json

    calib = _make_calib_json(np.eye(4, dtype=float))
    json_path = tmp_path / "calib_identity.json"
    json_path.write_text(json.dumps(calib))

    client = FakeXrClient()
    device = XRoboToolkitDevice(
        XRoboToolkitDeviceCfg(
            control_mode="relative",
            xr_client=client,
            delta_pos_axis_map=IDENTITY_AXIS_MAP,
            delta_rot_axis_map=IDENTITY_AXIS_MAP,
            calibration_json=str(json_path),
        )
    )

    client.keys["right_grip"] = 1.0
    _ = device.advance()

    client.pose = _pose([0.1, 0.2, -0.3], [0.1, 0.2, -0.3])
    action = device.advance()
    np.testing.assert_allclose(action.cpu().numpy()[:6], np.array([0.1, 0.2, -0.3, 0.1, 0.2, -0.3]), atol=1.0e-6)


def test_calibration_json_rot_map(tmp_path):
    """R_rot_map rotates the rotation delta after axis mapping."""
    import json

    # R_rot_map: swap X and Y, negate Z
    R_rot_map = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype=float)
    calib = _make_calib_json(np.eye(4, dtype=float), R_rot_map)
    json_path = tmp_path / "calib_rot_map.json"
    json_path.write_text(json.dumps(calib))

    client = FakeXrClient()
    device = XRoboToolkitDevice(
        XRoboToolkitDeviceCfg(
            control_mode="relative",
            xr_client=client,
            delta_pos_axis_map=IDENTITY_AXIS_MAP,
            delta_rot_axis_map=IDENTITY_AXIS_MAP,
            calibration_json=str(json_path),
        )
    )

    client.keys["right_grip"] = 1.0
    _ = device.advance()

    # Position delta should NOT be affected by R_rot_map
    client.pose = _pose([0.1, 0.0, 0.0])
    action = device.advance()
    np.testing.assert_allclose(action.cpu().numpy()[:3], np.array([0.1, 0.0, 0.0]), atol=1.0e-6)

    # Rotation delta SHOULD be mapped: [0.1, 0.2, -0.3] -> R_rot_map -> [0.2, 0.1, 0.3]
    client.pose = _pose([0.0, 0.0, 0.0], [0.1, 0.2, -0.3])
    action = device.advance()
    expected_rot = R_rot_map @ np.array([0.1, 0.2, -0.3])
    np.testing.assert_allclose(action.cpu().numpy()[3:6], expected_rot, atol=1.0e-6)


def test_calibration_json_axis_map(tmp_path):
    """Calibration W_T_Q matches default OPENXR_TO_ROS axis map."""
    import json

    openxr_to_ros = np.array(
        [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=float,
    )
    W_T_Q = np.eye(4, dtype=float)
    W_T_Q[:3, :3] = openxr_to_ros
    calib = _make_calib_json(W_T_Q)
    json_path = tmp_path / "calib_axis_map.json"
    json_path.write_text(json.dumps(calib))

    client = FakeXrClient()
    device = XRoboToolkitDevice(
        XRoboToolkitDeviceCfg(
            control_mode="relative",
            xr_client=client,
            calibration_json=str(json_path),
        )
    )

    client.keys["right_grip"] = 1.0
    _ = device.advance()

    # Same test as test_xrobotoolkit_device_default_ros_base_position_mapping
    client.pose = _pose([0.0, 0.0, -0.1])
    action = device.advance()
    np.testing.assert_allclose(action.cpu().numpy()[:3], np.array([0.1, 0.0, 0.0]), atol=1.0e-6)

    client.pose = _pose([-0.1, 0.0, 0.0])
    action = device.advance()
    np.testing.assert_allclose(action.cpu().numpy()[:3], np.array([0.0, 0.1, 0.0]), atol=1.0e-6)

    # Rotation mapping
    client.pose = _pose([0.0, 0.0, 0.0], [0.0, 0.0, -0.2])
    action = device.advance()
    np.testing.assert_allclose(action.cpu().numpy()[3:6], np.array([0.2, 0.0, 0.0]), atol=1.0e-6)


def test_axis_map_mode_ignores_calibration_json(tmp_path):
    """axis_map mode preserves the legacy hard-coded 3x3 mapping even when a JSON is provided."""
    import json

    calib = _make_calib_json(np.eye(4, dtype=float))
    json_path = tmp_path / "calib_identity.json"
    json_path.write_text(json.dumps(calib))

    client = FakeXrClient()
    device = XRoboToolkitDevice(
        XRoboToolkitDeviceCfg(
            control_mode="relative",
            mapping_mode="axis_map",
            xr_client=client,
            calibration_json=str(json_path),
        )
    )

    client.keys["right_grip"] = 1.0
    _ = device.advance()

    client.pose = _pose([0.0, 0.0, -0.1], [0.0, 0.0, -0.2])
    action = device.advance()
    np.testing.assert_allclose(
        action.cpu().numpy()[:6],
        np.array([0.1, 0.0, 0.0, 0.2, 0.0, 0.0]),
        atol=1.0e-6,
    )
