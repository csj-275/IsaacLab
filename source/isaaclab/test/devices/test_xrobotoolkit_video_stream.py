# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the XRoboToolkit video control and streaming helpers."""

from __future__ import annotations

import importlib.util
import socket
import struct
import sys
import time
from pathlib import Path

import numpy as np
import pytest


_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "isaaclab"
    / "devices"
    / "xrobotoolkit"
    / "xrobotoolkit_video_stream.py"
)
_SPEC = importlib.util.spec_from_file_location("xrobotoolkit_video_stream_test_module", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
video = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = video
_SPEC.loader.exec_module(video)


def _compact_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    assert len(encoded) <= 255
    return bytes([len(encoded)]) + encoded


def _camera_request(
    *,
    camera: str = "ZED",
    ip: str = "127.0.0.1",
    port: int = 12345,
    width: int = 4,
    height: int = 3,
    fps: int = 30,
    bitrate: int = 4_000_000,
    enable_mv_hevc: int = 0,
    render_mode: int = 0,
) -> bytes:
    return (
        bytes([0xCA, 0xFE, 1])
        + struct.pack("<7i", width, height, fps, bitrate, enable_mv_hevc, render_mode, port)
        + _compact_string(camera)
        + _compact_string(ip)
    )


def _network_body(command: str, data: bytes = b"") -> bytes:
    command_bytes = command.encode("utf-8")
    return struct.pack("<i", len(command_bytes)) + command_bytes + struct.pack("<i", len(data)) + data


def _framed_control(command: str, data: bytes = b"") -> bytes:
    body = _network_body(command, data)
    return struct.pack(">I", len(body)) + body


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError(f"expected {size} bytes, got {len(chunks)}")
        chunks.extend(chunk)
    return bytes(chunks)


def _wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class FakeEncoder:
    instances: list["FakeEncoder"] = []

    def __init__(self, request):
        self.request = request
        self.frames: list[np.ndarray] = []
        self.closed = False
        FakeEncoder.instances.append(self)

    def encode(self, image_rgb: np.ndarray) -> list[bytes]:
        self.frames.append(image_rgb.copy())
        return [b"encoded-frame"]

    def flush(self) -> list[bytes]:
        self.closed = True
        return []


def _receiver_socket() -> socket.socket:
    receiver = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    receiver.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    receiver.bind(("127.0.0.1", 0))
    receiver.listen(1)
    receiver.settimeout(2.0)
    return receiver


def test_parse_open_camera_protocol():
    protocol = video.parse_network_protocol(_network_body("OPEN_CAMERA", _camera_request(port=4567)))
    request = video.parse_camera_request(protocol.data)

    assert protocol.command == "OPEN_CAMERA"
    assert request.camera == "ZED"
    assert request.ip == "127.0.0.1"
    assert request.port == 4567
    assert request.width == 4
    assert request.height == 3
    assert request.fps == 30
    assert request.bitrate == 4_000_000


def test_parse_close_camera_protocol():
    protocol = video.parse_network_protocol(_network_body("CLOSE_CAMERA"))

    assert protocol.command == "CLOSE_CAMERA"
    assert protocol.data == b""


def test_pop_framed_protocols_waits_for_partial_body():
    first = _framed_control("OPEN_CAMERA", _camera_request())
    second = _framed_control("CLOSE_CAMERA")
    buffer = bytearray(first[:7])

    assert video.pop_framed_protocols(buffer) == []

    buffer.extend(first[7:] + second)
    protocols = video.pop_framed_protocols(buffer)

    assert [protocol.command for protocol in protocols] == ["OPEN_CAMERA", "CLOSE_CAMERA"]
    assert buffer == bytearray()


def test_reject_non_zed_camera_type():
    server = video.XRoboToolkitVideoStreamServer(
        "127.0.0.1:0",
        strict_camera_type="ZED",
        encoder_factory=FakeEncoder,
    )
    protocol = video.NetworkDataProtocol("OPEN_CAMERA", _camera_request(camera="D435"))

    with pytest.raises(video.ProtocolError, match="unsupported camera type"):
        server.handle_protocol(protocol)


def test_length_prefix_payload_uses_big_endian_size():
    packet = video.length_prefix_payload(b"abc")

    assert packet == b"\x00\x00\x00\x03abc"
    assert struct.unpack(">I", packet[:4])[0] == 3


def test_streamer_sends_encoded_payload_and_stops_on_close():
    FakeEncoder.instances.clear()
    frame = np.zeros((3, 4, 3), dtype=np.uint8)
    receiver = _receiver_socket()
    server = video.XRoboToolkitVideoStreamServer(
        "127.0.0.1:0",
        default_fps=60,
        encoder_factory=FakeEncoder,
    )
    stream_conn = None
    control = None

    try:
        server.start()
        control = socket.create_connection(server.bound_address, timeout=2.0)
        video_port = receiver.getsockname()[1]
        control.sendall(_framed_control("OPEN_CAMERA", _camera_request(port=video_port)))

        stream_conn, _ = receiver.accept()
        stream_conn.settimeout(2.0)
        assert _wait_until(lambda: server.is_streaming)

        server.submit_frame(frame)
        payload_len = struct.unpack(">I", _recv_exact(stream_conn, 4))[0]
        payload = _recv_exact(stream_conn, payload_len)

        assert payload == b"encoded-frame"
        assert FakeEncoder.instances[0].request.port == video_port
        np.testing.assert_array_equal(FakeEncoder.instances[0].frames[0], frame)

        control.sendall(_framed_control("CLOSE_CAMERA"))
        assert _wait_until(lambda: not server.is_streaming)
        assert FakeEncoder.instances[0].closed is True
    finally:
        if stream_conn is not None:
            stream_conn.close()
        if control is not None:
            control.close()
        receiver.close()
        server.stop()


def test_streamer_stops_when_video_receiver_resets_connection():
    FakeEncoder.instances.clear()
    frame = np.zeros((3, 4, 3), dtype=np.uint8)
    receiver = _receiver_socket()
    server = video.XRoboToolkitVideoStreamServer(
        "127.0.0.1:0",
        default_fps=60,
        encoder_factory=FakeEncoder,
    )
    stream_conn = None
    control = None

    try:
        server.start()
        control = socket.create_connection(server.bound_address, timeout=2.0)
        video_port = receiver.getsockname()[1]
        control.sendall(_framed_control("OPEN_CAMERA", _camera_request(port=video_port)))
        stream_conn, _ = receiver.accept()
        assert _wait_until(lambda: server.is_streaming)

        stream_conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        stream_conn.close()
        stream_conn = None

        for _ in range(20):
            server.submit_frame(frame)
            if _wait_until(lambda: not server.is_streaming, timeout=0.1):
                break

        assert not server.is_streaming
        assert FakeEncoder.instances[0].closed is True
    finally:
        if stream_conn is not None:
            stream_conn.close()
        if control is not None:
            control.close()
        receiver.close()
        server.stop()
