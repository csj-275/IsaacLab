# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""XRoboToolkit-compatible video streaming helpers."""

from __future__ import annotations

import logging
import queue
import socket
import struct
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_LISTEN_ADDRESS = "0.0.0.0:13579"
MAX_CONTROL_BODY_BYTES = 1_048_576
LOOPBACK_VIDEO_HOSTS = {"", "0.0.0.0", "127.0.0.1", "::1", "localhost"}


class ProtocolError(ValueError):
    """Raised when an XRoboToolkit video control message is malformed."""


@dataclass(frozen=True)
class CameraRequestData:
    """Decoded XRoboToolkit OPEN_CAMERA request payload."""

    width: int
    height: int
    fps: int
    bitrate: int
    enable_mv_hevc: int
    render_mode: int
    port: int
    camera: str
    ip: str


@dataclass(frozen=True)
class NetworkDataProtocol:
    """Decoded XRoboToolkit command envelope."""

    command: str
    data: bytes


class VideoEncoder(Protocol):
    """Encoder interface used by the video stream server."""

    def encode(self, image_rgb: np.ndarray) -> list[bytes]:
        """Encode one RGB frame into one or more payload chunks."""

    def flush(self) -> list[bytes]:
        """Flush delayed encoder payloads."""


EncoderFactory = Callable[[CameraRequestData], VideoEncoder]


def _read_i32_le(data: bytes, offset: int) -> int:
    if offset + 4 > len(data):
        raise ProtocolError("not enough bytes for int32")
    return struct.unpack_from("<i", data, offset)[0]


def _read_compact_string(data: bytes, offset: int) -> tuple[str, int]:
    if offset >= len(data):
        raise ProtocolError("not enough bytes for compact string length")

    length = data[offset]
    offset += 1
    if offset + length > len(data):
        raise ProtocolError("not enough bytes for compact string payload")

    try:
        value = data[offset : offset + length].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProtocolError("compact string is not valid UTF-8") from exc

    return value, offset + length


def parse_camera_request(data: bytes) -> CameraRequestData:
    """Parse XRoboToolkit ``CameraRequestData`` bytes."""
    if len(data) < 2 + 1 + 7 * 4 + 2:
        raise ProtocolError("camera request is too short")
    if data[0] != 0xCA or data[1] != 0xFE:
        raise ProtocolError("invalid camera request magic bytes")
    if data[2] != 1:
        raise ProtocolError(f"unsupported camera request version {data[2]}")

    offset = 3
    width = _read_i32_le(data, offset)
    height = _read_i32_le(data, offset + 4)
    fps = _read_i32_le(data, offset + 8)
    bitrate = _read_i32_le(data, offset + 12)
    enable_mv_hevc = _read_i32_le(data, offset + 16)
    render_mode = _read_i32_le(data, offset + 20)
    port = _read_i32_le(data, offset + 24)
    offset += 28

    camera, offset = _read_compact_string(data, offset)
    ip, offset = _read_compact_string(data, offset)
    if offset != len(data):
        raise ProtocolError(f"camera request has {len(data) - offset} trailing byte(s)")

    return CameraRequestData(
        width=width,
        height=height,
        fps=fps,
        bitrate=bitrate,
        enable_mv_hevc=enable_mv_hevc,
        render_mode=render_mode,
        port=port,
        camera=camera,
        ip=ip,
    )


def parse_network_protocol(body: bytes) -> NetworkDataProtocol:
    """Parse an XRoboToolkit ``NetworkDataProtocol`` body."""
    if len(body) < 8:
        raise ProtocolError("network protocol body is too short")

    offset = 0
    command_length = _read_i32_le(body, offset)
    offset += 4
    if command_length < 0 or offset + command_length > len(body):
        raise ProtocolError("invalid command length")

    try:
        command = body[offset : offset + command_length].split(b"\0", 1)[0].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProtocolError("command is not valid UTF-8") from exc
    offset += command_length

    if offset + 4 > len(body):
        raise ProtocolError("missing data length")
    data_length = _read_i32_le(body, offset)
    offset += 4
    if data_length < 0 or offset + data_length > len(body):
        raise ProtocolError("invalid data length")

    data = body[offset : offset + data_length]
    offset += data_length
    if offset != len(body):
        raise ProtocolError(f"network protocol has {len(body) - offset} trailing byte(s)")

    return NetworkDataProtocol(command=command, data=data)


def pop_framed_protocols(buffer: bytearray) -> list[NetworkDataProtocol]:
    """Pop all complete length-prefixed control messages from ``buffer``."""
    protocols: list[NetworkDataProtocol] = []
    while True:
        if len(buffer) < 4:
            return protocols

        body_length = struct.unpack_from(">I", buffer, 0)[0]
        if body_length > MAX_CONTROL_BODY_BYTES:
            raise ProtocolError(f"control body length {body_length} exceeds limit")
        if len(buffer) < 4 + body_length:
            return protocols

        body = bytes(buffer[4 : 4 + body_length])
        del buffer[: 4 + body_length]
        protocols.append(parse_network_protocol(body))


def length_prefix_payload(payload: bytes) -> bytes:
    """Prefix an encoded video payload with the XRoboToolkit 4-byte big-endian length."""
    return struct.pack(">I", len(payload)) + payload


def parse_host_port(address: str) -> tuple[str, int]:
    """Parse a ``HOST:PORT`` address."""
    if ":" not in address:
        raise ValueError("address must be in HOST:PORT format")
    host, port_text = address.rsplit(":", 1)
    if not host:
        raise ValueError("address host must not be empty")
    port = int(port_text)
    if port < 0 or port > 65535:
        raise ValueError(f"port out of range: {port}")
    return host, port


def _as_rgb_uint8(image_rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"expected HxWx3 or HxWx4 image, got shape {image.shape}")
    if image.shape[2] == 4:
        image = image[..., :3]
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def resize_rgb(image_rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize an RGB frame to the requested encoder dimensions."""
    image = _as_rgb_uint8(image_rgb)
    if image.shape[1] == width and image.shape[0] == height:
        return image

    try:
        import cv2

        return np.ascontiguousarray(cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA))
    except ModuleNotFoundError:
        from PIL import Image

        pil_image = Image.fromarray(image, mode="RGB")
        pil_image = pil_image.resize((width, height), Image.Resampling.BILINEAR)
        return np.ascontiguousarray(np.asarray(pil_image, dtype=np.uint8))


class _LatestFrameBuffer:
    """Single-slot frame handoff that drops old frames when the encoder falls behind."""

    def __init__(self):
        self._condition = threading.Condition()
        self._frame: np.ndarray | None = None
        self._sequence = 0
        self._closed = False

    def submit(self, image_rgb: np.ndarray) -> None:
        frame = _as_rgb_uint8(image_rgb).copy()
        with self._condition:
            self._frame = frame
            self._sequence += 1
            self._condition.notify_all()

    def wait_next(self, last_sequence: int, timeout: float = 0.5) -> tuple[int, np.ndarray | None]:
        with self._condition:
            self._condition.wait_for(lambda: self._closed or self._sequence != last_sequence, timeout=timeout)
            if self._closed:
                return self._sequence, None
            if self._sequence == last_sequence or self._frame is None:
                return last_sequence, None
            return self._sequence, self._frame

    def notify(self) -> None:
        with self._condition:
            self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()


class FFmpegVideoEncoder:
    """Low-latency H.264/HEVC encoder backed by the bundled imageio-ffmpeg binary."""

    def __init__(self, request: CameraRequestData):
        self.width = request.width
        self.height = request.height
        self.fps = request.fps if request.fps > 0 else 30
        self.bitrate = request.bitrate if request.bitrate > 0 else 4_000_000
        self.use_hevc = request.enable_mv_hevc != 0
        self._output_queue: queue.Queue[bytes] = queue.Queue()
        self._closed = False

        command = self._make_command()
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        self._reader_thread = threading.Thread(target=self._read_stdout, name="xrobotoolkit-ffmpeg-reader", daemon=True)
        self._reader_thread.start()
        logger.info(
            "started %s encoder for %dx%d@%d",
            "HEVC" if self.use_hevc else "H.264",
            self.width,
            self.height,
            self.fps,
        )

    def _make_command(self) -> list[str]:
        try:
            import imageio_ffmpeg
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "XRoboToolkit video streaming requires imageio_ffmpeg in the Isaac Lab environment."
            ) from exc

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        codec = "libx265" if self.use_hevc else "libx264"
        stream_format = "hevc" if self.use_hevc else "h264"
        params_name = "x265-params" if self.use_hevc else "x264-params"
        params_value = "keyint=15:min-keyint=15:scenecut=0:repeat-headers=1"

        return [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s:v",
            f"{self.width}x{self.height}",
            "-r",
            str(self.fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            codec,
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-b:v",
            str(self.bitrate),
            "-g",
            "15",
            "-bf",
            "0",
            f"-{params_name}",
            params_value,
            "-pix_fmt",
            "yuv420p",
            "-f",
            stream_format,
            "pipe:1",
        ]

    def _read_stdout(self) -> None:
        assert self._process.stdout is not None
        while True:
            chunk = self._process.stdout.read(64 * 1024)
            if not chunk:
                break
            self._output_queue.put(chunk)

    def _drain_output(self) -> list[bytes]:
        chunks: list[bytes] = []
        while True:
            try:
                chunks.append(self._output_queue.get_nowait())
            except queue.Empty:
                return chunks

    def encode(self, image_rgb: np.ndarray) -> list[bytes]:
        if self._closed:
            return []
        if self._process.poll() is not None:
            raise RuntimeError("ffmpeg encoder process exited")
        if self._process.stdin is None:
            raise RuntimeError("ffmpeg encoder stdin is unavailable")

        frame = resize_rgb(image_rgb, self.width, self.height)
        self._process.stdin.write(frame.tobytes())
        self._process.stdin.flush()
        return self._drain_output()

    def flush(self) -> list[bytes]:
        if self._closed:
            return []
        self._closed = True
        if self._process.stdin is not None:
            try:
                self._process.stdin.close()
            except OSError:
                pass
        try:
            self._process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=2.0)
        self._reader_thread.join(timeout=1.0)
        return self._drain_output()

    def __del__(self):
        try:
            self.flush()
        except Exception:
            pass


class XRoboToolkitVideoStreamServer:
    """Listen for XRoboToolkit camera commands and stream submitted RGB frames."""

    def __init__(
        self,
        listen_address: str = DEFAULT_LISTEN_ADDRESS,
        *,
        strict_camera_type: str = "ZED",
        default_width: int = 640,
        default_height: int = 480,
        default_fps: int = 30,
        default_bitrate: int = 4_000_000,
        encoder_factory: EncoderFactory = FFmpegVideoEncoder,
    ):
        self.listen_host, self.listen_port = parse_host_port(listen_address)
        self.strict_camera_type = strict_camera_type
        self.default_width = default_width
        self.default_height = default_height
        self.default_fps = default_fps
        self.default_bitrate = default_bitrate
        self.encoder_factory = encoder_factory
        self.bound_address: tuple[str, int] | None = None

        self._frame_buffer = _LatestFrameBuffer()
        self._server_socket: socket.socket | None = None
        self._control_thread: threading.Thread | None = None
        self._control_stop_event = threading.Event()
        self._stream_thread: threading.Thread | None = None
        self._stream_stop_event = threading.Event()
        self._lock = threading.Lock()

    @property
    def is_streaming(self) -> bool:
        """Whether a video stream thread is currently alive."""
        return self._stream_thread is not None and self._stream_thread.is_alive()

    def start(self) -> None:
        """Start listening for XRoboToolkit control commands."""
        if self._control_thread is not None and self._control_thread.is_alive():
            return

        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((self.listen_host, self.listen_port))
        server_socket.listen(1)
        server_socket.settimeout(0.2)

        self._server_socket = server_socket
        self.bound_address = server_socket.getsockname()
        self._control_stop_event.clear()
        self._control_thread = threading.Thread(
            target=self._serve_loop,
            name="xrobotoolkit-video-control",
            daemon=True,
        )
        self._control_thread.start()
        logger.info("listening for XRoboToolkit video commands on %s:%d", *self.bound_address)

    def stop(self) -> None:
        """Stop control listening and any active stream."""
        self._control_stop_event.set()
        self.stop_stream()
        if self._server_socket is not None:
            try:
                self._server_socket.close()
            except OSError:
                pass
            self._server_socket = None

        if self._control_thread is not None and self._control_thread.is_alive():
            self._control_thread.join(timeout=2.0)
        self._control_thread = None
        self._frame_buffer.notify()

    def submit_frame(self, image_rgb: np.ndarray) -> None:
        """Submit the latest RGB camera frame from the simulation thread."""
        if self.is_streaming:
            self._frame_buffer.submit(image_rgb)

    def stop_stream(self) -> None:
        """Stop only the active video stream."""
        with self._lock:
            thread = self._stream_thread
            if thread is None:
                return
            self._stream_stop_event.set()
            self._frame_buffer.notify()
            if thread.is_alive():
                thread.join(timeout=2.0)
            if not thread.is_alive():
                self._stream_thread = None

    def handle_protocol(self, protocol: NetworkDataProtocol, *, peer_host: str | None = None) -> None:
        """Handle a decoded XRoboToolkit control command."""
        if protocol.command == "OPEN_CAMERA":
            request = parse_camera_request(protocol.data)
            if self.strict_camera_type and request.camera != self.strict_camera_type:
                raise ProtocolError(
                    f"unsupported camera type {request.camera!r}; expected {self.strict_camera_type!r}"
                )
            self._start_stream(self._with_control_peer_host(self._with_defaults(request), peer_host))
        elif protocol.command == "CLOSE_CAMERA":
            logger.info("received CLOSE_CAMERA")
            print("XRoboToolkit CLOSE_CAMERA received")
            self.stop_stream()
        else:
            raise ProtocolError(f"unknown command {protocol.command!r}")

    def _with_defaults(self, request: CameraRequestData) -> CameraRequestData:
        return replace(
            request,
            width=request.width if request.width > 0 else self.default_width,
            height=request.height if request.height > 0 else self.default_height,
            fps=request.fps if request.fps > 0 else self.default_fps,
            bitrate=request.bitrate if request.bitrate > 0 else self.default_bitrate,
        )

    def _with_control_peer_host(self, request: CameraRequestData, peer_host: str | None) -> CameraRequestData:
        if peer_host is None or request.ip not in LOOPBACK_VIDEO_HOSTS or peer_host in LOOPBACK_VIDEO_HOSTS:
            return request

        logger.warning(
            "OPEN_CAMERA requested loopback video host %s; using control peer host %s instead",
            request.ip,
            peer_host,
        )
        print(f"XRoboToolkit OPEN_CAMERA requested loopback host {request.ip}; using {peer_host}")
        return replace(request, ip=peer_host)

    def _start_stream(self, request: CameraRequestData) -> None:
        if request.port <= 0 or request.port > 65535:
            raise ProtocolError(f"invalid video stream port: {request.port}")
        if request.width <= 0 or request.height <= 0:
            raise ProtocolError(f"invalid video stream size: {request.width}x{request.height}")
        if not request.ip:
            raise ProtocolError("empty video stream IP")

        self.stop_stream()
        with self._lock:
            self._stream_stop_event.clear()
            self._stream_thread = threading.Thread(
                target=self._stream_loop,
                args=(request,),
                name="xrobotoolkit-video-stream",
                daemon=True,
            )
            self._stream_thread.start()
        logger.info(
            "received OPEN_CAMERA: %dx%d@%d bitrate=%d target=%s:%d camera=%s",
            request.width,
            request.height,
            request.fps,
            request.bitrate,
            request.ip,
            request.port,
            request.camera,
        )
        print(
            "XRoboToolkit OPEN_CAMERA: "
            f"{request.width}x{request.height}@{request.fps} bitrate={request.bitrate} "
            f"target={request.ip}:{request.port} camera={request.camera}"
        )

    def _serve_loop(self) -> None:
        while not self._control_stop_event.is_set():
            try:
                assert self._server_socket is not None
                conn, addr = self._server_socket.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            logger.info("XRoboToolkit video control client connected from %s:%d", addr[0], addr[1])
            print(f"XRoboToolkit video control client connected from {addr[0]}:{addr[1]}")
            try:
                self._handle_client(conn, peer_host=addr[0])
            finally:
                self.stop_stream()
                logger.info("XRoboToolkit video control client disconnected")
                print("XRoboToolkit video control client disconnected")

    def _handle_client(self, conn: socket.socket, *, peer_host: str | None = None) -> None:
        buffer = bytearray()
        conn.settimeout(0.2)
        with conn:
            while not self._control_stop_event.is_set():
                try:
                    chunk = conn.recv(4096)
                except socket.timeout:
                    continue
                except OSError as exc:
                    logger.warning("XRoboToolkit video control receive failed: %s", exc)
                    break
                if not chunk:
                    break

                buffer.extend(chunk)
                try:
                    protocols = pop_framed_protocols(buffer)
                except ProtocolError as exc:
                    logger.warning("dropping malformed XRoboToolkit video control message: %s", exc)
                    buffer.clear()
                    continue

                for protocol in protocols:
                    try:
                        self.handle_protocol(protocol, peer_host=peer_host)
                    except ProtocolError as exc:
                        logger.warning("ignored XRoboToolkit video control command: %s", exc)

    def _stream_loop(self, request: CameraRequestData) -> None:
        encoder: VideoEncoder | None = None
        last_sequence = 0
        frame_interval_s = 1.0 / request.fps if request.fps > 0 else 1.0 / self.default_fps
        try:
            encoder = self.encoder_factory(request)
            with socket.create_connection((request.ip, request.port), timeout=5.0) as stream_socket:
                stream_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                logger.info("connected XRoboToolkit video stream to %s:%d", request.ip, request.port)
                print(f"XRoboToolkit video stream connected to {request.ip}:{request.port}")
                while not self._stream_stop_event.is_set():
                    loop_start = time.perf_counter()
                    last_sequence, frame = self._frame_buffer.wait_next(last_sequence, timeout=0.5)
                    if frame is None:
                        continue

                    frame = resize_rgb(frame, request.width, request.height)
                    for payload in encoder.encode(frame):
                        if payload:
                            stream_socket.sendall(length_prefix_payload(payload))

                    elapsed = time.perf_counter() - loop_start
                    time.sleep(max(0.0, frame_interval_s - elapsed))

                for payload in encoder.flush():
                    if payload:
                        stream_socket.sendall(length_prefix_payload(payload))
        except (BrokenPipeError, ConnectionResetError) as exc:
            logger.warning("XRoboToolkit video receiver closed the stream: %s", exc)
        except OSError as exc:
            logger.error("XRoboToolkit video stream socket error: %s", exc)
        except Exception as exc:
            logger.error("XRoboToolkit video stream failed: %s", exc)
        finally:
            if encoder is not None:
                try:
                    encoder.flush()
                except Exception:
                    pass
            logger.info("XRoboToolkit video stream stopped")
