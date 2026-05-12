#!/usr/bin/env bash
# Setup script for XRoboToolkit environment inside Isaac Lab Docker container.
#
# Prerequisites:
#   The container must be started with the xrobotoolkit patch overlay, which
#   bind-mounts the following host directories into /workspace:
#     external/xrobotoolkit/xrobotoolkit                    -> /workspace/xrobotoolkit
#     external/xrobotoolkit/XRoboToolkit-PC-Service-Pybind  -> /workspace/XRoboToolkit-PC-Service-Pybind
#     external/xrobotoolkit/XRoboToolkit-PC-Service         -> /workspace/XRoboToolkit-PC-Service
#
# Usage (inside the container):
#   bash /workspace/isaaclab/scripts/tools/setup_xrobotoolkit_env.sh
#
set -euo pipefail

ISAACLAB_SH="${ISAACLAB_PATH:-/workspace/isaaclab}/isaaclab.sh"

XR_PC_PYBIND="/workspace/XRoboToolkit-PC-Service-Pybind"
XR_PC_SERVICE="/workspace/XRoboToolkit-PC-Service"
XR_TELEOP="/workspace/xrobotoolkit"
SDK_BUILD_ROOT=""

cleanup() {
    if [ -n "$SDK_BUILD_ROOT" ]; then
        rm -rf "$SDK_BUILD_ROOT"
    fi
}
trap cleanup EXIT

echo "=== XRoboToolkit Environment Setup ==="

# Step 1: Build native libPXREARobotSDK.so from local PC-Service
echo "[1/4] Building native libPXREARobotSDK.so..."
if [ ! -d "$XR_PC_SERVICE" ]; then
    echo "ERROR: $XR_PC_SERVICE not found. Is the xrobotoolkit patch overlay active?"
    exit 1
fi

SDK_BUILD_ROOT="$(mktemp -d /tmp/xrobotoolkit_pc_service.XXXXXX)"
XR_PC_SERVICE_BUILD="$SDK_BUILD_ROOT/XRoboToolkit-PC-Service"
cp -a "$XR_PC_SERVICE" "$XR_PC_SERVICE_BUILD"
rm -rf "$XR_PC_SERVICE_BUILD/RoboticsService/PXREARobotSDK/build"

cd "$XR_PC_SERVICE_BUILD/RoboticsService/PXREARobotSDK"
bash build.sh
echo "  -> native SDK build complete"

# Step 2: Copy headers and shared library into pybind repo
echo "[2/4] Copying headers and library into pybind repo..."
mkdir -p "$XR_PC_PYBIND/include" "$XR_PC_PYBIND/lib"

cp "$XR_PC_SERVICE_BUILD/RoboticsService/PXREARobotSDK/PXREARobotSDK.h" "$XR_PC_PYBIND/include/"
cp -r "$XR_PC_SERVICE_BUILD/RoboticsService/PXREARobotSDK/nlohmann" "$XR_PC_PYBIND/include/"

SDK_BUILD_DIR="$XR_PC_SERVICE_BUILD/RoboticsService/PXREARobotSDK/build"
if [ -f "$SDK_BUILD_DIR/libPXREARobotSDK.so" ]; then
    cp "$SDK_BUILD_DIR/libPXREARobotSDK.so" "$XR_PC_PYBIND/lib/"
elif [ -f "$XR_PC_SERVICE_BUILD/RoboticsService/SDK/linux/64/libPXREARobotSDK.so" ]; then
    cp "$XR_PC_SERVICE_BUILD/RoboticsService/SDK/linux/64/libPXREARobotSDK.so" "$XR_PC_PYBIND/lib/"
else
    echo "Searching for libPXREARobotSDK.so..."
    SO_PATH=$(find "$XR_PC_SERVICE_BUILD" -name "libPXREARobotSDK.so" -print -quit 2>/dev/null)
    if [ -z "$SO_PATH" ]; then
        echo "ERROR: libPXREARobotSDK.so not found after build"
        exit 1
    fi
    cp "$SO_PATH" "$XR_PC_PYBIND/lib/"
fi

# Install shared library system-wide so the pybind11 extension can find it at runtime
cp "$XR_PC_PYBIND/lib/libPXREARobotSDK.so" /usr/local/lib/
ldconfig
echo "  -> headers and libPXREARobotSDK.so installed"

# Step 3: Build and install xrobotoolkit_sdk (pybind11 extension)
echo "[3/4] Installing xrobotoolkit_sdk..."
cd "$XR_PC_PYBIND"

$ISAACLAB_SH -p -m pip uninstall -y xrobotoolkit_sdk 2>/dev/null || true
if ! PYBIND11_CMAKE_DIR="$($ISAACLAB_SH -p -c "import pybind11; print(pybind11.get_cmake_dir())" | tail -n 1)"; then
    echo "ERROR: pybind11 is not available in Isaac Lab Python. Rebuild the image or install pybind11 first."
    exit 1
fi
if [ ! -f "$PYBIND11_CMAKE_DIR/pybind11Config.cmake" ]; then
    echo "ERROR: pybind11Config.cmake not found under $PYBIND11_CMAKE_DIR"
    exit 1
fi
echo "  -> pybind11 CMake dir: $PYBIND11_CMAKE_DIR"

# Remove stale CMake cache from a previous failed build before reinstalling.
rm -rf "$XR_PC_PYBIND/build" "$XR_PC_PYBIND"/*.egg-info

pybind11_DIR="$PYBIND11_CMAKE_DIR" \
CMAKE_PREFIX_PATH="${PYBIND11_CMAKE_DIR}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}" \
    $ISAACLAB_SH -p -m pip install --no-build-isolation --no-deps --force-reinstall .
echo "  -> xrobotoolkit_sdk installed"

# Step 4: Install xrobotoolkit_teleop (editable, no dependency resolution)
echo "[4/4] Installing xrobotoolkit_teleop (editable)..."
if [ ! -d "$XR_TELEOP" ]; then
    echo "ERROR: $XR_TELEOP not found. Is the xrobotoolkit patch overlay active?"
    exit 1
fi

$ISAACLAB_SH -p -m pip install --no-deps -e "$XR_TELEOP"
echo "  -> xrobotoolkit_teleop installed (editable)"

# Smoke test
echo ""
echo "=== Smoke tests ==="
$ISAACLAB_SH -p -c "import xrobotoolkit_sdk; print('  [OK] xrobotoolkit_sdk')"
$ISAACLAB_SH -p -c "from xrobotoolkit_teleop.common.xr_client import XrClient; print('  [OK] xrobotoolkit_teleop')"
TERM="${TERM:-xterm}" $ISAACLAB_SH -p - <<'PY'
from isaaclab.app import AppLauncher
import sys

simulation_app = AppLauncher(headless=True).app
try:
    from isaaclab.devices.xrobotoolkit import XRoboToolkitDeviceCfg

    sys.__stdout__.write("  [OK] isaaclab device\n")
    sys.__stdout__.flush()
finally:
    simulation_app.close()
PY

echo ""
echo "=== XRoboToolkit environment setup complete ==="
