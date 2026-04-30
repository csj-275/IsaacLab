# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the XRoboToolkit teleoperation device."""

from __future__ import annotations

import numpy as np
import torch

from isaaclab.devices.xrobotoolkit.xrobotoolkit_device import XRoboToolkitDevice, XRoboToolkitDeviceCfg


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

    client.pose = np.array([0.1, -0.2, 0.3, 0.0, 0.0, np.sin(np.pi / 4.0), np.cos(np.pi / 4.0)])
    client.keys["right_trigger"] = 1.0
    action = device.advance()
    np.testing.assert_allclose(
        action.cpu().numpy(),
        np.array([0.1, -0.2, 0.3, 0.0, 0.0, np.pi / 2.0, -1.0]),
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
