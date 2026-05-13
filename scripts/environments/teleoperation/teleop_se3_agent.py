# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run teleoperation with Isaac Lab manipulation environments.

Supports multiple input devices (e.g., keyboard, spacemouse, gamepad) and devices
configured within the environment (including OpenXR-based hand tracking or motion
controllers)."""

"""Launch Isaac Sim Simulator first."""

import argparse
from collections.abc import Callable

from isaaclab.app import AppLauncher

_XROBOT_DEFAULT_VIDEO_LISTEN = "0.0.0.0:13579"

# add argparse arguments
parser = argparse.ArgumentParser(description="Teleoperation for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    help=(
        "Teleop device. Set here (legacy) or via the environment config. If using the environment config, pass the"
        " device key/name defined under 'teleop_devices' (it can be a custom name, not necessarily 'handtracking')."
        " Built-ins: keyboard, spacemouse, gamepad, xrobotoolkit. Not all tasks support all built-ins."
    ),
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
parser.add_argument(
    "--xrobotoolkit_control_mode",
    "--xrobotoolkit-control-mode",
    choices=("relative", "absolute"),
    default=None,
    help="XRoboToolkit control mode. If omitted, the environment device config is used.",
)
parser.add_argument(
    "--xrobotoolkit_mapping_mode",
    "--xrobotoolkit-mapping-mode",
    choices=("world_frame_calibrated", "axis_map"),
    default=None,
    help="XRoboToolkit mapping mode. If omitted, the environment device config is used.",
)
parser.add_argument(
    "--xrobotoolkit_debug_mapping",
    "--xrobotoolkit-debug-mapping",
    action="store_true",
    default=False,
    help="Print XRoboToolkit raw and mapped controller deltas for coordinate calibration.",
)
parser.add_argument(
    "--xrobotoolkit_calibration_json",
    "--xrobotoolkit-calibration-json",
    type=str,
    default=None,
    help="Path to a piper world-frame calibration JSON file.",
)
parser.add_argument(
    "--disable_xrobotoolkit_video_stream",
    "--disable-xrobotoolkit-video-stream",
    action="store_true",
    default=False,
    help="Disable XRoboToolkit wrist camera video streaming.",
)
parser.add_argument(
    "--xrobotoolkit_video_listen",
    "--xrobotoolkit-video-listen",
    type=str,
    default=_XROBOT_DEFAULT_VIDEO_LISTEN,
    help="XRoboToolkit video control listen address in HOST:PORT format.",
)
parser.add_argument(
    "--xrobotoolkit_video_camera",
    "--xrobotoolkit-video-camera",
    type=str,
    default="wrist_cam",
    help="Isaac Lab scene camera entity used for XRoboToolkit video streaming.",
)
parser.add_argument(
    "--xrobotoolkit_video_width",
    "--xrobotoolkit-video-width",
    type=int,
    default=640,
    help="Default local XRoboToolkit video camera width in pixels.",
)
parser.add_argument(
    "--xrobotoolkit_video_height",
    "--xrobotoolkit-video-height",
    type=int,
    default=480,
    help="Default local XRoboToolkit video camera height in pixels.",
)
parser.add_argument(
    "--xrobotoolkit_video_fps",
    "--xrobotoolkit-video-fps",
    type=int,
    default=30,
    help="Default XRoboToolkit video output frame rate in frames per second.",
)
parser.add_argument(
    "--xrobotoolkit_video_bitrate",
    "--xrobotoolkit-video-bitrate",
    type=int,
    default=4_000_000,
    help="Default XRoboToolkit video output bitrate in bits per second.",
)
parser.add_argument(
    "--xrobotoolkit_video_strict_camera_type",
    "--xrobotoolkit-video-strict-camera-type",
    type=str,
    default="ZED",
    help="Only accept XRoboToolkit OPEN_CAMERA requests with this camera type. Empty string accepts all.",
)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)
parser.add_argument(
    "--show_camera_display",
    "--show-camera-display",
    action="store_true",
    default=False,
    help="Show a native omni.ui window displaying camera feeds side by side.",
)
parser.add_argument(
    "--camera_display_cameras",
    "--camera-display-cameras",
    type=str,
    nargs="+",
    default=["wrist_cam", "table_cam"],
    help="List of scene camera names to show in the camera display window.",
)
parser.add_argument(
    "--auto_table_cam_frustum",
    "--auto-table-cam-frustum",
    action="store_true",
    default=False,
    help="Override the configured table_cam pose with frustum-based automatic placement.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and
    # not the one installed by Isaac Sim pinocchio is required by the Pink IK controllers and the
    # GR1T2 retargeter
    import pinocchio  # noqa: F401
if "handtracking" in args_cli.teleop_device.lower():
    app_launcher_args["xr"] = True
if args_cli.teleop_device.lower() == "xrobotoolkit" and not args_cli.disable_xrobotoolkit_video_stream:
    app_launcher_args["enable_cameras"] = True
if args_cli.show_camera_display:
    app_launcher_args["enable_cameras"] = True

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

"""Rest everything follows."""


import logging
import math

import gymnasium as gym
import numpy as np
import torch

from isaaclab.devices import (
    Se3Gamepad,
    Se3GamepadCfg,
    Se3Keyboard,
    Se3KeyboardCfg,
    Se3SpaceMouse,
    Se3SpaceMouseCfg,
    XRoboToolkitDevice,
    XRoboToolkitDeviceCfg,
)
from isaaclab.devices.openxr import remove_camera_configs
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.devices.xrobotoolkit.xrobotoolkit_video_stream import XRoboToolkitVideoStreamServer
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import CameraCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.utils import parse_env_cfg

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401

# import logger
logger = logging.getLogger(__name__)


def _get_xrobotoolkit_cfg(env_cfg: ManagerBasedRLEnvCfg) -> XRoboToolkitDeviceCfg | None:
    if not hasattr(env_cfg, "teleop_devices"):
        return None
    device_cfg = env_cfg.teleop_devices.devices.get("xrobotoolkit")
    return device_cfg if isinstance(device_cfg, XRoboToolkitDeviceCfg) else None


def _configure_xrobotoolkit_control_mode(env_cfg: ManagerBasedRLEnvCfg) -> str:
    device_cfg = _get_xrobotoolkit_cfg(env_cfg)
    cfg_mode = getattr(device_cfg, "control_mode", "absolute") if device_cfg is not None else "absolute"
    control_mode = args_cli.xrobotoolkit_control_mode or cfg_mode
    if device_cfg is not None:
        device_cfg.control_mode = control_mode
        if args_cli.xrobotoolkit_mapping_mode:
            device_cfg.mapping_mode = args_cli.xrobotoolkit_mapping_mode
        if args_cli.xrobotoolkit_debug_mapping:
            device_cfg.debug_mapping = True
        if args_cli.xrobotoolkit_calibration_json:
            device_cfg.calibration_json = args_cli.xrobotoolkit_calibration_json

    arm_action = getattr(env_cfg.actions, "arm_action", None)
    if arm_action is None or not hasattr(arm_action, "controller"):
        raise ValueError("XRoboToolkit control mode requires env_cfg.actions.arm_action with an IK controller.")

    if control_mode == "absolute":
        if getattr(env_cfg.scene, "ee_frame", None) is None:
            raise ValueError("XRoboToolkit absolute mode requires env_cfg.scene.ee_frame.")
        arm_action.controller.use_relative_mode = False
        arm_action.scale = 1.0
    else:
        arm_action.controller.use_relative_mode = True

    return control_mode


def _make_ee_pose_provider(env: gym.Env):
    def _provider():
        try:
            ee_frame = env.scene["ee_frame"]
        except KeyError as exc:
            raise RuntimeError("XRoboToolkit absolute mode requires env.scene['ee_frame'].") from exc

        pos = ee_frame.data.target_pos_source[0, 0].detach().cpu().numpy()
        quat = ee_frame.data.target_quat_source[0, 0].detach().cpu().numpy()
        return pos, quat

    return _provider


def _attach_xrobotoolkit_pose_provider(device: object, env: gym.Env, control_mode: str | None):
    if control_mode == "absolute":
        if not isinstance(device, XRoboToolkitDevice):
            raise TypeError("XRoboToolkit absolute mode requires an XRoboToolkitDevice instance.")
        device.set_absolute_pose_provider(_make_ee_pose_provider(env))


def _xrobotoolkit_video_requested() -> bool:
    return args_cli.teleop_device.lower() == "xrobotoolkit" and not args_cli.disable_xrobotoolkit_video_stream


def _configure_xrobotoolkit_video_camera(env_cfg: ManagerBasedRLEnvCfg) -> bool:
    if not _xrobotoolkit_video_requested():
        return False

    camera_name = args_cli.xrobotoolkit_video_camera
    camera_cfg = getattr(env_cfg.scene, camera_name, None)
    if not isinstance(camera_cfg, CameraCfg):
        message = f"XRoboToolkit video camera '{camera_name}' is not a CameraCfg scene entity."
        if camera_name == "wrist_cam":
            logger.warning("%s Video streaming will be disabled.", message)
            return False
        raise ValueError(message)

    camera_cfg.width = args_cli.xrobotoolkit_video_width
    camera_cfg.height = args_cli.xrobotoolkit_video_height
    camera_cfg.update_period = 0.0
    data_types = list(camera_cfg.data_types or [])
    if "rgb" not in data_types:
        data_types.append("rgb")
    camera_cfg.data_types = data_types
    return True


def _create_xrobotoolkit_video_stream(enabled: bool) -> XRoboToolkitVideoStreamServer | None:
    if not enabled:
        return None

    strict_camera_type = args_cli.xrobotoolkit_video_strict_camera_type or ""
    server = XRoboToolkitVideoStreamServer(
        listen_address=args_cli.xrobotoolkit_video_listen,
        strict_camera_type=strict_camera_type,
        default_width=args_cli.xrobotoolkit_video_width,
        default_height=args_cli.xrobotoolkit_video_height,
        default_fps=args_cli.xrobotoolkit_video_fps,
        default_bitrate=args_cli.xrobotoolkit_video_bitrate,
    )
    try:
        server.start()
    except OSError as exc:
        logger.error("Failed to start XRoboToolkit video control listener: %s", exc)
        return None

    host, port = server.bound_address if server.bound_address is not None else (server.listen_host, server.listen_port)
    print(f"XRoboToolkit video control listening on {host}:{port}")
    return server


def _submit_xrobotoolkit_video_frame(
    video_stream: XRoboToolkitVideoStreamServer | None,
    env: gym.Env,
    *,
    force_camera_update: bool,
) -> None:
    if video_stream is None or not video_stream.is_streaming:
        return

    try:
        camera = env.scene[args_cli.xrobotoolkit_video_camera]
        if force_camera_update:
            camera.update(env.sim.get_physics_dt(), force_recompute=True)
        rgb = camera.data.output.get("rgb")
        if rgb is None:
            logger.warning("XRoboToolkit video camera '%s' has no RGB output.", args_cli.xrobotoolkit_video_camera)
            return

        frame = rgb[0].detach().cpu().numpy() if hasattr(rgb, "detach") else np.asarray(rgb)[0]
        video_stream.submit_frame(frame)
    except Exception as exc:
        logger.warning("Failed to submit XRoboToolkit video frame: %s", exc)


def _get_table_camera_fit_prims(env: gym.Env) -> tuple[list[str], list[str]]:
    """Collect scene prims that should be visible in the table camera."""
    excluded_names = {"terrain", "plane", "ground", "table_cam", "wrist_cam"}
    excluded_tokens = ("camera", "cam", "frame", "marker", "light")
    prim_paths: list[str] = []
    entity_names: list[str] = []

    for name in env.scene.keys():
        lower_name = name.lower()
        if lower_name in excluded_names or any(token in lower_name for token in excluded_tokens):
            continue

        try:
            entity = env.scene[name]
        except KeyError:
            continue

        prim_path = _get_entity_fit_prim_path(entity)
        if prim_path is None:
            continue

        # Use env_0 for fitting. The resulting camera pose is shifted to other env origins below.
        prim_paths.append(prim_path)
        entity_names.append(name)

    return prim_paths, entity_names


def _get_entity_fit_prim_path(entity) -> str | None:
    """Resolve the first concrete prim path for a scene entity."""
    entity_prim_paths = getattr(entity, "prim_paths", None)
    if entity_prim_paths:
        return entity_prim_paths[0]

    cfg = getattr(entity, "cfg", None)
    cfg_prim_path = getattr(cfg, "prim_path", None)
    if cfg_prim_path:
        try:
            import isaaclab.sim as sim_utils

            matching_paths = sim_utils.find_matching_prim_paths(cfg_prim_path)
            if matching_paths:
                return matching_paths[0]
        except Exception as exc:
            logger.debug("Could not resolve scene entity cfg prim path '%s': %s", cfg_prim_path, exc)

    root_physx_view = getattr(entity, "root_physx_view", None)
    root_prim_paths = getattr(root_physx_view, "prim_paths", None)
    if root_prim_paths:
        return root_prim_paths[0]

    return None


def _get_scene_aligned_range(prim_paths: list[str]):
    """Compute a combined world-space aligned range for the given prim subtrees."""
    import omni.usd
    from pxr import Gf, Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()
    included_purposes = [
        getattr(UsdGeom.Tokens, "default_", "default"),
        getattr(UsdGeom.Tokens, "render", "render"),
        getattr(UsdGeom.Tokens, "proxy", "proxy"),
    ]
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), included_purposes, True)
    overall_min = Gf.Vec3d(float("inf"))
    overall_max = Gf.Vec3d(float("-inf"))
    found = False

    for path in prim_paths:
        try:
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                continue
            rng = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
            if rng.IsEmpty():
                continue
            found = True
            for i in range(3):
                overall_min[i] = min(overall_min[i], rng.GetMin()[i])
                overall_max[i] = max(overall_max[i], rng.GetMax()[i])
        except Exception as exc:
            logger.debug("Skipping prim '%s' during table_cam AABB fit: %s", path, exc)

    if not found:
        return None
    return overall_min, overall_max


def _get_camera_sensor_prims(camera) -> list:
    """Return the UsdGeom.Camera prims owned by an Isaac Lab Camera sensor."""
    sensor_prims = getattr(camera, "_sensor_prims", None)
    if sensor_prims:
        return sensor_prims

    prim_paths = getattr(camera, "prim_paths", None)
    if not prim_paths:
        return []

    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    camera_prims = []
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if prim and prim.IsValid() and prim.IsA(UsdGeom.Camera):
            camera_prims.append(UsdGeom.Camera(prim))
    return camera_prims


def _check_points_in_camera_frustum(
    camera,
    points_world: torch.Tensor,
    *,
    eye: np.ndarray | None = None,
    target: np.ndarray | None = None,
    h_fov: float | None = None,
    v_fov: float | None = None,
    clipping_range: tuple[float, float] | None = None,
) -> bool:
    """Check if all world-space points are visible in the camera's view frustum.

    Args:
        camera: An initialized Camera sensor.
        points_world: World-space points to check, shape (N, 3).

    Returns:
        True if all points project to NDC within [-1, 1].
    """
    if eye is not None and target is not None and h_fov is not None and v_fov is not None:
        eye_t = torch.tensor(eye, dtype=torch.float64)
        target_t = torch.tensor(target, dtype=torch.float64)
        points_t = points_world.to(dtype=torch.float64)

        forward = torch.nn.functional.normalize(target_t - eye_t, dim=0, eps=1e-12)
        up_axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
        right = torch.cross(forward, up_axis, dim=0)
        if torch.linalg.norm(right) < 1e-8:
            up_axis = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
            right = torch.cross(forward, up_axis, dim=0)
        right = torch.nn.functional.normalize(right, dim=0, eps=1e-12)
        up = torch.nn.functional.normalize(torch.cross(right, forward, dim=0), dim=0, eps=1e-12)

        rel = points_t - eye_t
        depth = rel @ forward
        x = rel @ right
        y = rel @ up
        near_clip, far_clip = clipping_range if clipping_range is not None else (0.0, float("inf"))
        if not bool(torch.all((depth > near_clip) & (depth < far_clip))):
            return False

        max_x = depth * math.tan(h_fov * 0.5)
        max_y = depth * math.tan(v_fov * 0.5)
        return bool(torch.all(x.abs() <= max_x) and torch.all(y.abs() <= max_y))

    from pxr import Usd

    camera_prims = _get_camera_sensor_prims(camera)
    if not camera_prims:
        return False

    gf_cam = camera_prims[0].GetCamera(Usd.TimeCode.Default())
    view_mat = torch.tensor(np.array(gf_cam.frustum.ComputeViewMatrix()), dtype=torch.float64).reshape(4, 4)
    proj_mat = torch.tensor(np.array(gf_cam.frustum.ComputeProjectionMatrix()), dtype=torch.float64).reshape(4, 4)

    n = points_world.shape[0]
    pts_h = torch.cat([points_world.double(), torch.ones(n, 1, dtype=torch.float64)], dim=1)
    clip = (proj_mat @ view_mat @ pts_h.T).T
    valid_w = clip[:, 3].abs() > 1e-12
    if not bool(torch.all(valid_w)):
        return False
    ndc = clip[:, :3] / clip[:, 3:4]
    return bool(torch.all(ndc.abs() <= 1.0))


def _compute_scene_aabb(prim_paths: list[str]):
    """Compute the combined world-space AABB for the given prim paths.

    Returns:
        Tuple of (center: np.ndarray[3], half_extents: np.ndarray[3], radius: float) or None.
    """
    result = _get_scene_aligned_range(prim_paths)
    if result is None:
        return None

    overall_min, overall_max = result
    center = np.array([(overall_min[i] + overall_max[i]) * 0.5 for i in range(3)])
    half_extents = np.array([(overall_max[i] - overall_min[i]) * 0.5 for i in range(3)])
    return center, half_extents, float(np.linalg.norm(half_extents))


def _range_points(mn: list[float], mx: list[float]) -> list[list[float]]:
    """Return center and corners for an aligned 3D range."""
    points = [[(mn[i] + mx[i]) * 0.5 for i in range(3)]]
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                points.append([mn[i] if v == 0 else mx[i] for i, v in enumerate((dx, dy, dz))])
    return points


def _points_aabb(points: torch.Tensor) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute center, half extents, and radius from world-space points."""
    points_np = points.cpu().numpy()
    points_min = points_np.min(axis=0)
    points_max = points_np.max(axis=0)
    center = (points_min + points_max) * 0.5
    half_extents = (points_max - points_min) * 0.5
    return center, half_extents, float(np.linalg.norm(half_extents))


def _scene_check_points(prim_paths: list[str], entity_names: list[str]) -> torch.Tensor | None:
    """Get world-space points that table_cam must observe.

    The table contributes only top-surface workspace samples. Robot and cube
    entities contribute full AABB samples and remain hard visibility targets.

    Returns:
        Tensor of shape (N, 3) or None.
    """
    table_range = None
    required_points: list[list[float]] = []

    for path, name in zip(prim_paths, entity_names):
        result = _get_scene_aligned_range([path])
        if result is None:
            continue
        aligned_min, aligned_max = result
        mn = [aligned_min[i] for i in range(3)]
        mx = [aligned_max[i] for i in range(3)]
        if name.lower() == "table":
            table_range = (mn, mx)
        else:
            required_points.extend(_range_points(mn, mx))

    if table_range is not None:
        table_min, table_max = table_range
        table_top_z = table_max[2]
        if required_points:
            required_np = np.array(required_points)
            workspace_min_xy = required_np[:, :2].min(axis=0)
            workspace_max_xy = required_np[:, :2].max(axis=0)
            workspace_size_xy = workspace_max_xy - workspace_min_xy
            workspace_margin = max(float(workspace_size_xy.max()) * 0.15, 0.08)
            surface_min_xy = np.maximum(workspace_min_xy - workspace_margin, np.array(table_min[:2]))
            surface_max_xy = np.minimum(workspace_max_xy + workspace_margin, np.array(table_max[:2]))
        else:
            table_center_xy = (np.array(table_min[:2]) + np.array(table_max[:2])) * 0.5
            table_half_xy = (np.array(table_max[:2]) - np.array(table_min[:2])) * 0.25
            surface_min_xy = table_center_xy - table_half_xy
            surface_max_xy = table_center_xy + table_half_xy

        surface_center_xy = (surface_min_xy + surface_max_xy) * 0.5
        required_points.append([surface_center_xy[0], surface_center_xy[1], table_top_z])
        for x in (surface_min_xy[0], surface_max_xy[0]):
            for y in (surface_min_xy[1], surface_max_xy[1]):
                required_points.append([x, y, table_top_z])

    points = required_points
    if not points:
        return None
    return torch.tensor(points, dtype=torch.float32)


def _set_table_camera_view(env: gym.Env, table_cam, eye: np.ndarray, target: np.ndarray) -> None:
    """Set the table camera look-at pose for all cloned env cameras."""
    eyes = torch.tensor([eye.tolist()], dtype=torch.float32, device=env.device)
    targets = torch.tensor([target.tolist()], dtype=torch.float32, device=env.device)

    camera_count = len(_get_camera_sensor_prims(table_cam))
    if camera_count > 1 and hasattr(env.scene, "env_origins"):
        origins = env.scene.env_origins[:camera_count].to(device=env.device, dtype=torch.float32)
        origin_offsets = origins - origins[0:1]
        eyes = eyes.repeat(camera_count, 1) + origin_offsets
        targets = targets.repeat(camera_count, 1) + origin_offsets

    table_cam.set_world_poses_from_view(eyes=eyes, targets=targets)


def _setup_table_cam_with_frustum(env: gym.Env, table_cam, table_cam_cfg) -> None:
    """Position the table camera using a frustum-based search.

    Fits the camera around the combined scene AABB, orients it toward the
    workspace center, and verifies the final pose with the USD camera frustum.

    Args:
        env: The simulation environment.
        table_cam: An initialized Camera sensor for the table view.
        table_cam_cfg: The CameraCfg used to create table_cam.
    """
    paths, fit_entity_names = _get_table_camera_fit_prims(env)

    if not paths:
        logger.warning("No scene prims found; skipping frustum-based table_cam setup.")
        return

    points = _scene_check_points(paths, fit_entity_names)
    if points is None:
        logger.warning("Could not compute scene check points; skipping frustum-based table_cam setup.")
        return
    scene_center, scene_half_extents, scene_radius = _points_aabb(points)

    # Camera intrinsics from config
    focal_length = table_cam_cfg.spawn.focal_length
    h_aperture = table_cam_cfg.spawn.horizontal_aperture
    v_aperture = table_cam_cfg.spawn.vertical_aperture
    width = table_cam_cfg.width
    height = table_cam_cfg.height
    if v_aperture is None:
        v_aperture = h_aperture * height / width
    h_fov = 2.0 * math.atan(h_aperture / (2.0 * focal_length))
    v_fov = 2.0 * math.atan(v_aperture / (2.0 * focal_length))
    min_half_fov = min(h_fov, v_fov) * 0.5
    if min_half_fov <= 1e-6:
        logger.warning("Invalid table_cam FOV; skipping frustum-based setup.")
        return

    target = scene_center.copy()
    base_distance = max(scene_radius / math.sin(min_half_fov), 0.5) * 1.08
    near_clip, far_clip = table_cam_cfg.spawn.clipping_range
    if base_distance + scene_radius > far_clip:
        logger.warning(
            "table_cam far clipping range %.3f m may be too small for scene radius %.3f m and distance %.3f m.",
            far_clip,
            scene_radius,
            base_distance,
        )

    candidate_azimuths = (-90.0, -45.0, -135.0, 0.0, 180.0, 45.0, 135.0, 90.0)
    candidate_elevations = (30.0, 40.0, 50.0)
    distance_scales = (1.0, 1.2, 1.5, 2.0)

    for distance_scale in distance_scales:
        distance = base_distance * distance_scale
        if distance - scene_radius < near_clip:
            distance = near_clip + scene_radius + 0.05
        for elevation_deg in candidate_elevations:
            elevation = math.radians(elevation_deg)
            cos_elevation = math.cos(elevation)
            for azimuth_deg in candidate_azimuths:
                azimuth = math.radians(azimuth_deg)
                direction = np.array(
                    [
                        math.cos(azimuth) * cos_elevation,
                        math.sin(azimuth) * cos_elevation,
                        math.sin(elevation),
                    ],
                    dtype=np.float64,
                )
                eye = target + direction * distance
                eye[2] = max(eye[2], scene_center[2] + scene_half_extents[2] + 0.05)

                _set_table_camera_view(env, table_cam, eye, target)
                if _check_points_in_camera_frustum(
                    table_cam,
                    points,
                    eye=eye,
                    target=target,
                    h_fov=h_fov,
                    v_fov=v_fov,
                    clipping_range=table_cam_cfg.spawn.clipping_range,
                ):
                    print(
                        "Table camera placed via frustum: "
                        f"entities={fit_entity_names}, eye={eye.round(3).tolist()}, "
                        f"target={target.round(3).tolist()}, visible=True"
                    )
                    return

    fallback_eye = target + np.array([-0.7, -1.0, 0.7], dtype=np.float64) * base_distance * 1.5
    fallback_eye[2] = max(fallback_eye[2], scene_center[2] + scene_half_extents[2] + 0.05)
    _set_table_camera_view(env, table_cam, fallback_eye, target)
    if _check_points_in_camera_frustum(
        table_cam,
        points,
        eye=fallback_eye,
        target=target,
        h_fov=h_fov,
        v_fov=v_fov,
        clipping_range=table_cam_cfg.spawn.clipping_range,
    ):
        print(
            "Table camera placed via frustum fallback: "
            f"entities={fit_entity_names}, eye={fallback_eye.round(3).tolist()}, "
            f"target={target.round(3).tolist()}, visible=True"
        )
    else:
        logger.warning(
            "Table camera fallback did not fully contain scene AABB. entities=%s eye=%s target=%s",
            fit_entity_names,
            fallback_eye.round(3).tolist(),
            target.round(3).tolist(),
        )


def _setup_camera_viewports(env: gym.Env, camera_names: list[str]) -> None:
    """Bind scene cameras to Isaac Sim viewports.

    The first camera is bound to the active viewport (Viewport 1).
    Additional cameras each get a new viewport window.
    """
    try:
        import omni.kit.viewport.utility as vu
    except ImportError:
        logger.warning("omni.kit.viewport.utility not available; camera viewports disabled.")
        return

    for i, cam_name in enumerate(camera_names):
        try:
            camera = env.scene[cam_name]
            prim_path = camera._sensor_prims[0].GetPath().pathString
        except Exception as exc:
            logger.warning("Cannot bind camera '%s' to viewport: %s", cam_name, exc)
            continue

        if i == 0:
            vp = vu.get_active_viewport()
            if vp is not None:
                vp.camera_path = prim_path
                print(f"Viewport 1 bound to '{cam_name}' ({prim_path})")
        else:
            try:
                vp_window = vu.create_viewport_window(name=f"Camera: {cam_name}", width=640, height=480)
                vp = vp_window.viewport_api
                vp.camera_path = prim_path
                print(f"Viewport '{cam_name}' bound to {prim_path}")
            except Exception as exc:
                logger.warning("Failed to create viewport for camera '%s': %s", cam_name, exc)


def main() -> None:
    """
    Run teleoperation with an Isaac Lab manipulation environment.

    Creates the environment, sets up teleoperation interfaces and callbacks,
    and runs the main simulation loop until the application is closed.

    Returns:
        None
    """
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.env_name = args_cli.task
    if not isinstance(env_cfg, ManagerBasedRLEnvCfg):
        raise ValueError(
            "Teleoperation is only supported for ManagerBasedRLEnv environments. "
            f"Received environment config type: {type(env_cfg).__name__}"
        )
    # modify configuration
    env_cfg.terminations.time_out = None
    if "Lift" in args_cli.task:
        # set the resampling time range to large number to avoid resampling
        env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
        # add termination condition for reaching the goal otherwise the environment won't reset
        env_cfg.terminations.object_reached_goal = DoneTerm(func=mdp.object_reached_goal)

    xrobotoolkit_control_mode = None
    xrobotoolkit_video_enabled = False
    if args_cli.teleop_device.lower() == "xrobotoolkit":
        xrobotoolkit_control_mode = _configure_xrobotoolkit_control_mode(env_cfg)
        xrobotoolkit_video_enabled = _configure_xrobotoolkit_video_camera(env_cfg)

    if args_cli.xr:
        if xrobotoolkit_video_enabled:
            logger.warning("XR mode removes scene camera configs; XRoboToolkit video streaming will be disabled.")
            xrobotoolkit_video_enabled = False
        env_cfg = remove_camera_configs(env_cfg)
        env_cfg.sim.render.antialiasing_mode = "DLSS"

    try:
        # create environment
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        # check environment name (for reach , we don't allow the gripper)
        if "Reach" in args_cli.task:
            logger.warning(
                f"The environment '{args_cli.task}' does not support gripper control. The device command will be"
                " ignored."
            )
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        simulation_app.close()
        return

    # Flags for controlling teleoperation flow
    should_reset_recording_instance = False
    teleoperation_active = True

    # Callback handlers
    def reset_recording_instance() -> None:
        """
        Reset the environment to its initial state.

        Sets a flag to reset the environment on the next simulation step.

        Returns:
            None
        """
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True
        print("Reset triggered - Environment will reset on next step")

    def start_teleoperation() -> None:
        """
        Activate teleoperation control of the robot.

        Enables the application of teleoperation commands to the environment.

        Returns:
            None
        """
        nonlocal teleoperation_active
        teleoperation_active = True
        print("Teleoperation activated")

    def stop_teleoperation() -> None:
        """
        Deactivate teleoperation control of the robot.

        Disables the application of teleoperation commands to the environment.

        Returns:
            None
        """
        nonlocal teleoperation_active
        teleoperation_active = False
        print("Teleoperation deactivated")

    # Create device config if not already in env_cfg
    teleoperation_callbacks: dict[str, Callable[[], None]] = {
        "R": reset_recording_instance,
        "START": start_teleoperation,
        "STOP": stop_teleoperation,
        "RESET": reset_recording_instance,
    }

    # Devices with explicit START/STOP events begin inactive.
    if args_cli.xr or args_cli.teleop_device.lower() == "xrobotoolkit":
        teleoperation_active = False
    else:
        # Always active for other devices
        teleoperation_active = True

    # Create teleop device from config if present, otherwise create manually
    teleop_interface = None
    try:
        if hasattr(env_cfg, "teleop_devices") and args_cli.teleop_device in env_cfg.teleop_devices.devices:
            teleop_interface = create_teleop_device(
                args_cli.teleop_device, env_cfg.teleop_devices.devices, teleoperation_callbacks
            )
        else:
            logger.warning(
                f"No teleop device '{args_cli.teleop_device}' found in environment config. Creating default."
            )
            # Create fallback teleop device
            sensitivity = args_cli.sensitivity
            if args_cli.teleop_device.lower() == "keyboard":
                teleop_interface = Se3Keyboard(
                    Se3KeyboardCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.05 * sensitivity)
                )
            elif args_cli.teleop_device.lower() == "spacemouse":
                teleop_interface = Se3SpaceMouse(
                    Se3SpaceMouseCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.05 * sensitivity)
                )
            elif args_cli.teleop_device.lower() == "gamepad":
                teleop_interface = Se3Gamepad(
                    Se3GamepadCfg(pos_sensitivity=0.1 * sensitivity, rot_sensitivity=0.1 * sensitivity)
                )
            elif args_cli.teleop_device.lower() == "xrobotoolkit":
                teleop_interface = XRoboToolkitDevice(
                    XRoboToolkitDeviceCfg(
                        pos_sensitivity=sensitivity,
                        rot_sensitivity=sensitivity,
                        control_mode=xrobotoolkit_control_mode or "absolute",
                        mapping_mode=args_cli.xrobotoolkit_mapping_mode or "world_frame_calibrated",
                        debug_mapping=args_cli.xrobotoolkit_debug_mapping,
                        calibration_json=args_cli.xrobotoolkit_calibration_json,
                    )
                )
            else:
                logger.error(f"Unsupported teleop device: {args_cli.teleop_device}")
                logger.error("Supported devices: keyboard, spacemouse, gamepad, handtracking, xrobotoolkit")
                env.close()
                simulation_app.close()
                return

            # Add callbacks to fallback device
            for key, callback in teleoperation_callbacks.items():
                try:
                    teleop_interface.add_callback(key, callback)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to add callback for key {key}: {e}")

        _attach_xrobotoolkit_pose_provider(teleop_interface, env, xrobotoolkit_control_mode)
    except Exception as e:
        logger.error(f"Failed to create teleop device: {e}")
        env.close()
        simulation_app.close()
        return

    if teleop_interface is None:
        logger.error("Failed to create teleop interface")
        env.close()
        simulation_app.close()
        return

    print(f"Using teleop device: {teleop_interface}")

    # reset environment
    env.reset()
    teleop_interface.reset()

    # Optionally override the configured table camera pose using frustum-based search.
    if args_cli.auto_table_cam_frustum and "table_cam" in env.scene.keys() and hasattr(env_cfg.scene, "table_cam"):
        table_cam = env.scene["table_cam"]
        _setup_table_cam_with_frustum(env, table_cam, env_cfg.scene.table_cam)

    print("Teleoperation started. Press 'R' to reset the environment.")

    video_stream = _create_xrobotoolkit_video_stream(xrobotoolkit_video_enabled)

    # Bind scene cameras to viewports if requested
    if args_cli.show_camera_display:
        _setup_camera_viewports(env, args_cli.camera_display_cameras)

    # simulate environment
    try:
        while simulation_app.is_running():
            try:
                # run everything in inference mode
                with torch.inference_mode():
                    # get device command
                    action = teleop_interface.advance()

                    # Only apply teleop commands when active
                    if teleoperation_active:
                        # process actions
                        actions = action.repeat(env.num_envs, 1)
                        # apply actions
                        env.step(actions)
                    else:
                        env.sim.render()

                    _submit_xrobotoolkit_video_frame(
                        video_stream,
                        env,
                        force_camera_update=not teleoperation_active,
                    )

                    if should_reset_recording_instance:
                        env.reset()
                        teleop_interface.reset()
                        should_reset_recording_instance = False
                        print("Environment reset complete")
            except Exception as e:
                logger.error(f"Error during simulation step: {e}")
                break
    finally:
        if video_stream is not None:
            video_stream.stop()
        # close the simulator
        env.close()
        print("Environment closed")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
