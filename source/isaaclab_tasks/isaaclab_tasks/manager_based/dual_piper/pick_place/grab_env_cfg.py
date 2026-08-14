# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.spawners.materials.visual_materials_cfg import PreviewSurfaceCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp

# Paths
_BOX_USD_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../../../../../usd/box/box.usd")
)
_MUG_USD_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../../../../../usd/bottle/bottle.usd")
)

# ---------------------------------------------------------------------------
# Helper: auto-generate _left / _right observation term pairs
# ---------------------------------------------------------------------------


def _dual_obs_terms(specs: list[tuple[str, callable, dict, dict]]) -> dict[str, ObsTerm]:
    """Return a flat dict of ``{name}_left`` / ``{name}_right`` :class:`ObsTerm` pairs.

    Each element in *specs* is::

        (base_name, func, left_params, right_params)

    where *left_params* and *right_params* are the keyword-argument dicts passed
    to :class:`ObsTerm` for the left and right variants respectively.

    Example::

        _dual_obs_terms([
            ("eef_pos", mdp.ee_frame_pos,
             {"ee_frame_cfg": SceneEntityCfg("ee_frame_left")},
             {"ee_frame_cfg": SceneEntityCfg("ee_frame_right")}),
        ])
        # → {"eef_pos_left": ObsTerm(...), "eef_pos_right": ObsTerm(...)}
    """
    terms: dict[str, ObsTerm] = {}
    for name, func, left_p, right_p in specs:
        terms[f"{name}_left"] = ObsTerm(func=func, params=left_p)
        terms[f"{name}_right"] = ObsTerm(func=func, params=right_p)
    return terms


# Shortcuts so the table below stays compact.
_R_LEFT = SceneEntityCfg("robot_left")
_R_RIGHT = SceneEntityCfg("robot_right")
_EE_LEFT = SceneEntityCfg("ee_frame_left")
_EE_RIGHT = SceneEntityCfg("ee_frame_right")

# Shared specs reused by both PolicyCfg and SubtaskCfg.
_SHARED_SPECS: list[tuple[str, callable, dict, dict]] = [
    ("joint_pos", mdp.joint_pos_rel, {"asset_cfg": _R_LEFT}, {"asset_cfg": _R_RIGHT}),
    ("joint_vel", mdp.joint_vel_rel, {"asset_cfg": _R_LEFT}, {"asset_cfg": _R_RIGHT}),
    ("eef_pos", mdp.ee_frame_pos, {"ee_frame_cfg": _EE_LEFT}, {"ee_frame_cfg": _EE_RIGHT}),
    ("eef_quat", mdp.ee_frame_quat, {"ee_frame_cfg": _EE_LEFT}, {"ee_frame_cfg": _EE_RIGHT}),
    ("gripper_pos", mdp.gripper_pos, {"robot_cfg": _R_LEFT}, {"robot_cfg": _R_RIGHT}),
]

_OBJECT_SPECS: list[tuple[str, callable, dict, dict]] = [
    (
        "object_1_positions",
        mdp.object_poses_in_base_frame,
        {"object_cfg": SceneEntityCfg("object_1"), "robot_cfg": _R_LEFT, "return_key": "pos"},
        {"object_cfg": SceneEntityCfg("object_1"), "robot_cfg": _R_RIGHT, "return_key": "pos"},
    ),
    (
        "object_1_orientations",
        mdp.object_poses_in_base_frame,
        {"object_cfg": SceneEntityCfg("object_1"), "robot_cfg": _R_LEFT, "return_key": "quat"},
        {"object_cfg": SceneEntityCfg("object_1"), "robot_cfg": _R_RIGHT, "return_key": "quat"},
    ),
    (
        "box_positions",
        mdp.object_poses_in_base_frame,
        {"object_cfg": SceneEntityCfg("box"), "robot_cfg": _R_LEFT, "return_key": "pos"},
        {"object_cfg": SceneEntityCfg("box"), "robot_cfg": _R_RIGHT, "return_key": "pos"},
    ),
    (
        "box_orientations",
        mdp.object_poses_in_base_frame,
        {"object_cfg": SceneEntityCfg("box"), "robot_cfg": _R_LEFT, "return_key": "quat"},
        {"object_cfg": SceneEntityCfg("box"), "robot_cfg": _R_RIGHT, "return_key": "quat"},
    ),
]

_MUG_SPECS: list[tuple[str, callable, dict, dict]] = [
    (
        "mug_positions",
        mdp.object_poses_in_base_frame,
        {"object_cfg": SceneEntityCfg("mug"), "robot_cfg": _R_LEFT, "return_key": "pos"},
        {"object_cfg": SceneEntityCfg("mug"), "robot_cfg": _R_RIGHT, "return_key": "pos"},
    ),
    (
        "mug_orientations",
        mdp.object_poses_in_base_frame,
        {"object_cfg": SceneEntityCfg("mug"), "robot_cfg": _R_LEFT, "return_key": "quat"},
        {"object_cfg": SceneEntityCfg("mug"), "robot_cfg": _R_RIGHT, "return_key": "quat"},
    ),
]


# ---------------------------------------------------------------------------
# Scene definition
# ---------------------------------------------------------------------------

@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for a dual-arm tabletop scene.

    Derived classes must set the target robots, end-effector frames, and objects.
    """

    # robots: populated by agent env cfg
    robot_left: ArticulationCfg = MISSING
    robot_right: ArticulationCfg = MISSING
    # end-effector sensors: populated by agent env cfg
    ee_frame_left: FrameTransformerCfg = MISSING
    ee_frame_right: FrameTransformerCfg = MISSING

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.3, 0, 0), rot=(0.707, 0, 0, 0.707)),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
            scale=(1, 0.6, 1),
        ),
    )

    # Ground plane (invisible physics-only — warehouse provides visual floor)
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, -1.05)),
        spawn=UsdFileCfg(
            usd_path=os.path.join(os.path.dirname(__file__), "../../../../../../usd/ground/ground.usda"),
            rigid_props=RigidBodyPropertiesCfg(disable_gravity=True),
        ),
    )

    # Warehouse environment background (static, no physics)
    warehouse = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Warehouse",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, -1.05)),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd"),
    )


# --------------------------------------------------------------------
# MDP settings
# --------------------------------------------------------------------

@configclass
class ActionsCfg:
    """Action specifications for the dual-arm MDP."""

    # populated by agent env cfg
    arm_action_left: mdp.JointPositionActionCfg = MISSING
    arm_action_right: mdp.JointPositionActionCfg = MISSING
    gripper_action_left: mdp.BinaryJointPositionActionCfg = MISSING
    gripper_action_right: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the dual-arm MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group (state)."""

        actions = ObsTerm(func=mdp.last_action)

        # Object observations (world-frame, shared across arms; uses left ee as reference)
        object = ObsTerm(func=mdp.object_obs, params={"ee_frame_cfg": _EE_LEFT})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

            # Auto-generate all _left / _right observation pairs
            for k, v in _dual_obs_terms(_SHARED_SPECS + _OBJECT_SPECS + _MUG_SPECS).items():
                setattr(self, k, v)

    @configclass
    class RGBCameraPolicyCfg(ObsGroup):
        """Observations for policy group with RGB images."""

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for two-stage subtask group (cube→box, mug→box)."""

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

            for side, robot_cfg, ee_cfg in [
                ("left", _R_LEFT, _EE_LEFT),
                ("right", _R_RIGHT, _EE_RIGHT),
            ]:
                # grasp cube
                setattr(self, f"grasp_cube_{side}", ObsTerm(
                    func=mdp.object_grasped,
                    params={
                        "robot_cfg": robot_cfg,
                        "ee_frame_cfg": ee_cfg,
                        "object_cfg": SceneEntityCfg("object_1"),
                        "diff_threshold": 0.05,
                    },
                ))
                # placed cube into box
                setattr(self, f"placed_cube_{side}", ObsTerm(
                    func=mdp.object_a_is_into_b,
                    params={
                        "robot_cfg": robot_cfg,
                        "object_a_cfg": SceneEntityCfg("object_1"),
                        "object_b_cfg": SceneEntityCfg("box"),
                        "xy_threshold": 0.05,
                        "height_diff": 0.0,
                        "height_threshold": 0.05,
                        "gripper_threshold": 0.03,
                    },
                ))
                # grasp mug
                setattr(self, f"grasp_mug_{side}", ObsTerm(
                    func=mdp.object_grasped,
                    params={
                        "robot_cfg": robot_cfg,
                        "ee_frame_cfg": ee_cfg,
                        "object_cfg": SceneEntityCfg("mug"),
                        "diff_threshold": 0.05,
                    },
                ))

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class TerminationsCfg:
    """Termination terms for the dual-arm MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_1_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object_1")},
    )

    mug_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("mug")},
    )

    box_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("box")},
    )

    # Success: both object_1 (cube) and mug placed into box, gripper released
    success = DoneTerm(
        func=mdp.objects_a_and_b_are_into_c,
        params={
            "robot_cfg": SceneEntityCfg("robot_left"),
            "object_a_cfg": SceneEntityCfg("object_1"),
            "object_b_cfg": SceneEntityCfg("mug"),
            "object_c_cfg": SceneEntityCfg("box"),
        },
    )


@configclass
class EventCfg:
    """Configuration for dual-arm reset events (objects)."""

    # Combined cube + mug + box randomization in a single randomize_object_pose
    # call with per-object pose ranges and a min_separation constraint
    # (reference: piper_grab/V1).
    randomize_cube_and_mug_and_box_poses = EventTerm(
        func=mdp.randomize_object_pose,
        mode="reset",
        params={
            "pose_ranges": [
                {"x": (0.3, 0.35), "y": (0.1, 0.15), "z": (0.0203, 0.0203), "yaw": (-0.785, 0.785)},  # cube
                {"x": (0.2, 0.25), "y": (0.1, 0.15), "z": (0.0000, 0.0000), "yaw": (-0.785, 0.785)},  # mug
                {"x": (0.15, 0.2), "y": (0.25, 0.3), "z": (0.0000, 0.0000), "yaw": (-0.785, 0.785)},  # box
            ],
            "min_separation": 0.12,
            "asset_cfgs": [SceneEntityCfg("object_1"), SceneEntityCfg("mug"), SceneEntityCfg("box")],
        },
    )


# --------------------------------------------------------------------
# Environment configuration
# --------------------------------------------------------------------

@configclass
class GrabEnvCfg(ManagerBasedRLEnvCfg):
    """Base configuration for the dual-arm grabbing environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(
        num_envs=4096, env_spacing=2.5, replicate_physics=False
    )
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    # Unused managers
    rewards = None

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 5
        self.episode_length_s = 40.0
        # simulation settings
        self.sim.dt = 1 / 150
        self.sim.render_interval = 3

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.friction_correlation_distance = 0.00625
        # Reduced GPU memory allocations (warehouse scene consumes significant VRAM)
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 8 * 1024
        self.sim.physx.gpu_max_rigid_contact_count = 4 * 1024 * 1024
        self.sim.physx.gpu_heap_capacity = 32 * 1024 * 1024
        self.sim.physx.gpu_collision_stack_size = 32 * 1024 * 1024

        # Shared box (target container)
        self.scene.box = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/box",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.1, 0.3, 0.0203), rot=(1.0, 0.0, 0.0, 0.0)),
            spawn=UsdFileCfg(
                usd_path=_BOX_USD_PATH,
                rigid_props=RigidBodyPropertiesCfg(),
                semantic_tags=[("class", "box")],
            ),
        )

        # Shared mug (pickable object, e.g. bottle)
        _mug_props = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )
        self.scene.mug = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/mug",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, -0.15, 0.203), rot=(1, 0, 0, 0)),
            spawn=UsdFileCfg(
                usd_path=_MUG_USD_PATH,
                scale=(1, 1, 1),
                rigid_props=_mug_props,
                semantic_tags=[("class", "mug")],
            ),
        )

        # Shared cube (pickable blue block — reuses _mug_props, identical settings)
        self.scene.object_1 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/object_1",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, 0.0, 0.0203), rot=(1, 0, 0, 0)),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=_mug_props,
                semantic_tags=[("class", "cube_1")],
            ),
        )

        # ---- Fixed visual materials--------
        self.scene.mug.spawn.visual_material = PreviewSurfaceCfg(
            diffuse_color=(0.85, 0.45, 0.10),  # orange
            roughness=0.4,
            metallic=0.0,
        )
        self.scene.box.spawn.visual_material = PreviewSurfaceCfg(
            diffuse_color=(0.55, 0.35, 0.15),  # brown/cardboard
            roughness=0.5,
            metallic=0.0,
        )

        # ------------------------------------------------------------
        # Scene semantics
        # ------------------------------------------------------------
        self.scene.table.spawn.semantic_tags = [("class", "table")]
        self.scene.plane.spawn.semantic_tags = [("class", "ground")]
