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
    "--xrobotoolkit_debug_mapping",
    "--xrobotoolkit-debug-mapping",
    action="store_true",
    default=False,
    help="Print XRoboToolkit raw and mapped controller deltas for coordinate calibration.",
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

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

"""Rest everything follows."""


import logging

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
        if args_cli.xrobotoolkit_debug_mapping:
            device_cfg.debug_mapping = True

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
                        debug_mapping=args_cli.xrobotoolkit_debug_mapping,
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

    print("Teleoperation started. Press 'R' to reset the environment.")

    video_stream = _create_xrobotoolkit_video_stream(xrobotoolkit_video_enabled)

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
