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


def test_xrobotoolkit_device_control_edges_and_action_mapping():
    client = FakeXrClient()
    device = XRoboToolkitDevice(XRoboToolkitDeviceCfg(xr_client=client))
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

    client.pose = _pose([0.0, 0.0, -0.1], [np.pi / 2.0, 0.0, 0.0])
    client.keys["right_trigger"] = 1.0
    action = device.advance()
    np.testing.assert_allclose(
        action.cpu().numpy(),
        np.array([0.1, 0.0, 0.0, 0.0, -np.pi / 2.0, 0.0, -1.0]),
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


def test_xrobotoolkit_device_default_openxr_axis_mapping():
    client = FakeXrClient()
    device = XRoboToolkitDevice(XRoboToolkitDeviceCfg(xr_client=client))

    client.keys["right_grip"] = 1.0
    _ = device.advance()

    client.pose = _pose([0.1, 0.0, 0.0])
    action = device.advance()
    np.testing.assert_allclose(action.cpu().numpy()[:3], np.array([0.0, -0.1, 0.0]), atol=1.0e-6)

    client.pose = _pose([0.0, 0.1, 0.0])
    action = device.advance()
    np.testing.assert_allclose(action.cpu().numpy()[:3], np.array([0.0, 0.0, 0.1]), atol=1.0e-6)

    client.pose = _pose([0.0, 0.0, -0.1])
    action = device.advance()
    np.testing.assert_allclose(action.cpu().numpy()[:3], np.array([0.1, 0.0, 0.0]), atol=1.0e-6)

    client.pose = _pose([0.0, 0.0, 0.0], [0.1, 0.2, -0.3])
    action = device.advance()
    np.testing.assert_allclose(action.cpu().numpy()[3:6], np.array([0.3, -0.1, 0.2]), atol=1.0e-6)


def test_xrobotoolkit_device_axis_mapping_can_be_overridden():
    client = FakeXrClient()
    device = XRoboToolkitDevice(
        XRoboToolkitDeviceCfg(
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
