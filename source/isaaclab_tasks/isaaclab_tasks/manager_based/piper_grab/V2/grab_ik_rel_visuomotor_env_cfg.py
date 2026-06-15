# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""V2 visuomotor config — inherits V1, adds distractor cube + second mug."""

from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from isaaclab_tasks.manager_based.piper_grab import mdp
from isaaclab_tasks.manager_based.piper_grab.mdp import piper_grab_events
from isaaclab_tasks.manager_based.piper_grab.V1.grab_ik_rel_visuomotor_env_cfg import (
    PiperGrabVisuomotorEnvCfg as V1PiperGrabVisuomotorEnvCfg,
)


@configclass
class PiperGrabVisuomotorEnvCfg(V1PiperGrabVisuomotorEnvCfg):
    """V2: V1 + black distractor cube + second mug (from IsaacLab nucleus)."""

    def __post_init__(self):
        super().__post_init__()

        cube_props = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )

        # Add a black distractor cube
        self.scene.distractor = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/distractor",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.35, 0.08, 0.0203), rot=(1, 0, 0, 0)),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=cube_props,
                semantic_tags=[("class", "distractor")],
            ),
        )

        # Add a second mug from IsaacLab nucleus (smaller)
        self.scene.mug_2 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/mug_2",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.35, -0.1, 0.0003), rot=(1, 0, 0, 0)),
            spawn=UsdFileCfg(
                usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Objects/Mug/mug.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=cube_props,
                semantic_tags=[("class", "mug_2")],
            ),
        )

        # Replace V1's pose randomization with combined 5-object version
        del self.events.randomize_cube_and_mug_and_box_poses
        self.events.randomize_all_five_objects = EventTerm(
            func=piper_grab_events.randomize_object_pose,
            mode="reset",
            params={
                "pose_ranges": [
                    {"x": (0.20, 0.38), "y": (-0.12, 0.15), "z": (0.0203, 0.0203), "yaw": (-1.0, 1.0)},  # cube
                    {"x": (0.20, 0.38), "y": (-0.12, 0.15), "z": (0.0000, 0.0000), "yaw": (-1.0, 1.0)},  # mug
                    {"x": (0.08, 0.30), "y": (0.05, 0.35), "z": (0.0000, 0.0000), "yaw": (-1.0, 1.0)},  # box
                    {"x": (0.20, 0.40), "y": (-0.20, 0.10), "z": (0.0203, 0.0203), "yaw": (-1.0, 1.0)},  # distractor
                    {"x": (0.20, 0.40), "y": (-0.20, 0.10), "z": (0.0203, 0.0203), "yaw": (-1.0, 1.0)},  # mug_2
                ],
                "min_separation": 0.10,
                "asset_cfgs": [
                    SceneEntityCfg("object_1"),
                    SceneEntityCfg("mug"),
                    SceneEntityCfg("box"),
                    SceneEntityCfg("distractor"),
                    SceneEntityCfg("mug_2"),
                ],
            },
        )

        # Distractor color (dark)
        self.events.randomize_distractor_color = EventTerm(
            func=mdp.randomize_visual_color,
            mode="reset",
            params={
                "event_name": "randomize_distractor_color",
                "asset_cfg": SceneEntityCfg("distractor"),
                "colors": {"r": (0.05, 0.1), "g": (0.05, 0.1), "b": (0.05, 0.1)},
                "mesh_name": "",
            },
        )
