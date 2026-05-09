# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Isaac Lab is NVIDIA's GPU-accelerated robotics research framework built on NVIDIA Isaac Sim. It provides reinforcement learning, imitation learning, and motion planning environments with physics and sensor simulation.

Version: 2.3.2. Python 3.11. Line length: 120.

## Environment

This project uses Docker. All development happens inside the container.

```bash
# Start the container (with xrobotoolkit patch overlay)
./docker/container.py start base --files docker-compose.xrobotoolkit.patch.yaml

# Enter the running container
./docker/container.py enter base
```

The container mounts the repo at `/workspace/isaaclab`. Code is hot-reloaded — edits on the host are immediately visible inside.

## Essential commands (run inside the container)

Inside the container, `isaaclab` is an alias for `./isaaclab.sh` at the repo root.

```bash
# Install all extensions + RL frameworks
isaaclab --install

# Formatting and linting (ruff linter + formatter via pre-commit)
isaaclab --format

# Testing
isaaclab --test                                              # run all tests
isaaclab --test -k "test_name"                               # run specific test
isaaclab --test source/isaaclab/test/devices/test_xrobotoolkit_device.py

# Run a Python script within the Isaac Sim Python environment
isaaclab --python path/to/script.py

# Launch Isaac Sim
isaaclab --sim

# Build docs
isaaclab --docs
```

## Extension architecture

The codebase is organized as a collection of pip-installable extensions under `source/`, each following the same layout: `setup.py`, `docs/`, `test/`, `config/`, and the package directory.

| Extension | Purpose |
|---|---|
| `isaaclab` | Core framework: envs, managers, devices, assets, sensors, controllers, sim, terrains, markers, UI |
| `isaaclab_assets` | Robot/scene asset data files (USD, STL, DAE meshes) |
| `isaaclab_contrib` | Community-contributed actuators, MDP terms, sensors, utilities |
| `isaaclab_rl` | RL library wrappers (Stable Baselines 3, SKRL) |
| `isaaclab_mimic` | Imitation learning with Mimic |
| `isaaclab_tasks` | Task environment definitions — both `manager_based` and `direct` RL envs |

The `isaaclab_rl` extension supports optional framework installs: `isaaclab_rl[sb3]`, `isaaclab_rl[skrl]`, etc.

## Core framework (`source/isaaclab/isaaclab/`)

**Environment system** (`envs/`): Two families — `ManagerBasedRLEnv` and `DirectRLEnv`. Manager-based envs delegate to pluggable managers; direct envs give the user full control over the step loop.

**Manager pattern** (`managers/`): Modular components that compose RL environments:
- `ActionManager` — processes actions before applying to the sim
- `ObservationManager` — constructs observation dicts from the sim state
- `RewardManager` — computes reward terms
- `TerminationManager` — checks episode termination conditions
- `CommandManager` — generates goal commands (e.g., target velocities)
- `CurriculumManager` — adapts environment parameters over training
- `EventManager` — applies domain randomization and resets
- `RecorderManager` — logs rollout data

Each manager is configured via its own `*TermCfg` dataclass and registered as part of the env config.

**Devices** (`devices/`): Input device abstractions for teleoperation — `Gamepad`, `Keyboard`, `SpaceMouse`, `OpenXR`, `Haply`, `XRoboToolkit` (this branch).

**Config system**: Uses `omni.isaac.lab.utils.configclass` (a `dataclass`-like with `MISSING` sentinel). Task configs live in `source/isaaclab_tasks/isaaclab_tasks/{manager_based,direct}/`.

## This branch (`reconva`)

Adds XRoboToolkit teleoperation device support, Piper robot arm assets with camera, teleop SE(3) agent script, calibration tooling, and video streaming integration. Key files:
- `source/isaaclab/isaaclab/devices/xrobotoolkit/` — device driver and video stream
- `scripts/environments/teleoperation/teleop_se3_agent.py` — SE(3) teleop agent
- `scripts/tools/calibrate_xrobotoolkit.py` — calibration utility
- `scripts/tools/record_demos.py` — demonztration recording
- `docs/source/how-to/xrobotoolkit_teleoperation/` — user documentation

## Code style

- Ruff with `line-length=120`, Google-style docstrings (not enforced by lint)
- Imports follow isort section order: standard → third-party → omniverse → isaaclab extensions → local
- Type checking with pyright (basic mode), `reportMissingImports=none`, `reportGeneralTypeIssues=none`
- Use `torch` tensor operations for GPU-accelerated computation; avoid Python loops in hot paths
- Config classes use `dataclass`-style with `MISSING` sentinel for required fields
