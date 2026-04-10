# Newton SDF & Hydroelastic Config Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port SDF collision and hydroelastic shape preparation from PR #5160 into PR #5219's config design, as a new PR built on top of #5219's branch.

**Architecture:** Add `SDFCfg` configclass for SDF mesh preparation (patterns, resolution, hydroelastic flags). Add manager methods to build SDF on matching shapes before model finalization. Integrate with the Newton cloner to apply SDF on prototypes before replication.

**Tech Stack:** Python, Newton physics engine, Warp, IsaacLab configclass system

---

### Task 0: Create feature branch

**Files:** None

- [ ] **Step 1: Create branch off PR #5219**

```bash
git checkout expose_newton_collision_pipeline
git checkout -b antoiner/newton-sdf-config
```

- [ ] **Step 2: Commit the spec and plan docs**

```bash
git add docs/superpowers/specs/2026-04-10-newton-sdf-hydroelastic-config-design.md docs/superpowers/plans/2026-04-10-newton-sdf-hydroelastic.md
git commit -m "Add SDF/hydroelastic config design spec and implementation plan"
```

---

### Task 1: Add `SDFCfg` configclass

**Files:**
- Modify: `source/isaaclab_newton/isaaclab_newton/physics/newton_collision_cfg.py`

- [ ] **Step 1: Add `SDFCfg` class after `NewtonCollisionPipelineCfg`**

Add this class at the end of `newton_collision_cfg.py`:

```python
@configclass
class SDFCfg:
    """Configuration for SDF (Signed Distance Field) collision on Newton meshes.

    When provided as :attr:`~isaaclab_newton.physics.NewtonCfg.sdf_cfg`, mesh
    collision shapes matching the configured patterns will have SDF built via
    Newton's ``mesh.build_sdf()`` API before model finalization.

    At least one of :attr:`max_resolution` or :attr:`target_voxel_size` must be
    set for SDF to be built. At least one of :attr:`body_patterns` or
    :attr:`shape_patterns` must be set to select which shapes receive SDF.

    .. note::
        For hydroelastic contacts to be generated, shapes must have SDF built
        and the ``HYDROELASTIC`` flag set. Set :attr:`k_hydro` to enable
        hydroelastic on all matched shapes, or use
        :attr:`hydroelastic_shape_patterns` to limit which shapes get the flag.
        The pipeline-level hydroelastic processing parameters are configured
        separately via
        :attr:`NewtonCollisionPipelineCfg.sdf_hydroelastic_config`.
    """

    max_resolution: int | None = None
    """Maximum dimension [voxels] for sparse SDF grid (must be divisible by 8).

    Typical values: 128, 256, 512.
    """

    target_voxel_size: float | None = None
    """Target voxel size [m] for sparse SDF grid.

    If provided, takes precedence over :attr:`max_resolution`.
    """

    narrow_band_range: tuple[float, float] = (-0.1, 0.1)
    """Narrow band distance range (inner, outer) [m] for SDF computation."""

    margin: float | None = None
    """Collision margin [m] for SDF shapes. If ``None``, uses the builder's default."""

    body_patterns: list[str] | None = None
    """Regex patterns to match body labels (USD prim paths) for SDF.

    Bodies whose label matches at least one pattern will have SDF applied to
    all their mesh shapes. Example: ``[".*elbow.*", ".*wrist.*"]``.
    """

    shape_patterns: list[str] | None = None
    """Regex patterns to match shape labels (USD prim paths) for SDF.

    Only shapes whose label matches at least one pattern get SDF.
    Example: ``[".*Gear.*", ".*gear.*"]``.

    .. note::
        At least one of :attr:`body_patterns` or :attr:`shape_patterns` must
        be set for SDF to be applied.
    """

    pattern_resolutions: dict[str, int] | None = None
    """Per-pattern SDF resolution overrides.

    Maps regex pattern to ``max_resolution`` for matching shapes. Shapes not
    matching any pattern use the global :attr:`max_resolution`. First matching
    pattern wins. Example: ``{".*elbow.*": 128, ".*power_supply.*": 512}``.
    """

    use_visual_meshes: bool = False
    """Whether to create collision shapes from visual meshes for matched bodies
    that lack collision geometry.

    When ``False`` (default), only existing collision meshes are patched with
    SDF. When ``True``, bodies matching the configured patterns but lacking
    collision shapes get a new collision shape created from their first visual
    mesh.
    """

    k_hydro: float | None = None
    """Hydroelastic stiffness coefficient [Pa] applied to matched shapes.

    If ``None`` (default), the ``HYDROELASTIC`` flag is not set on any shapes.
    If set, matched shapes (optionally filtered by
    :attr:`hydroelastic_shape_patterns`) get the ``HYDROELASTIC`` flag and
    this stiffness value.

    .. note::
        Pipeline-level hydroelastic processing parameters (contact reduction,
        buffer sizes, etc.) are configured separately via
        :attr:`NewtonCollisionPipelineCfg.sdf_hydroelastic_config`.
    """

    hydroelastic_shape_patterns: list[str] | None = None
    """Regex patterns to select which SDF shapes also get hydroelastic contacts.

    If ``None`` and :attr:`k_hydro` is set, all shapes matching the SDF
    patterns get hydroelastic. If provided, only shapes whose label matches at
    least one pattern get the ``HYDROELASTIC`` flag.
    """
```

- [ ] **Step 2: Add missing pipeline params to `HydroelasticSDFCfg`**

Add these fields to `HydroelasticSDFCfg`, after the existing `output_contact_surface` field:

```python
    moment_matching: bool = False
    """Whether to adjust reduced contact friction so net maximum moment matches
    the unreduced reference.

    Only active when ``reduce_contacts`` is True.

    Defaults to ``False`` (same as Newton's default).
    """

    buffer_mult_broad: int = 1
    """Multiplier for preallocated broadphase buffer.

    Increase if a broadphase overflow warning is issued.

    Defaults to ``1`` (same as Newton's default).
    """

    buffer_mult_iso: int = 1
    """Multiplier for preallocated iso-surface extraction buffers.

    Increase if an iso buffer overflow warning is issued.

    Defaults to ``1`` (same as Newton's default).
    """

    buffer_mult_contact: int = 1
    """Multiplier for the preallocated face contact buffer.

    Increase if a face contact overflow warning is issued.

    Defaults to ``1`` (same as Newton's default).
    """

    grid_size: int = 262144
    """Grid size for hydroelastic contact handling.

    Defaults to ``262144`` (``256 * 8 * 128``, same as Newton's default).
    """
```

- [ ] **Step 3: Run pre-commit**

```bash
./isaaclab.sh -f
```

Expected: All checks pass.

- [ ] **Step 4: Commit**

```bash
git add source/isaaclab_newton/isaaclab_newton/physics/newton_collision_cfg.py
git commit -m "Add SDFCfg configclass and missing HydroelasticSDFCfg fields"
```

---

### Task 2: Wire `SDFCfg` into `NewtonCfg` and exports

**Files:**
- Modify: `source/isaaclab_newton/isaaclab_newton/physics/newton_manager_cfg.py`
- Modify: `source/isaaclab_newton/isaaclab_newton/physics/__init__.pyi`

- [ ] **Step 1: Add `sdf_cfg` field to `NewtonCfg`**

In `newton_manager_cfg.py`, add this import at the top (after the existing `NewtonCollisionPipelineCfg` import):

```python
from .newton_collision_cfg import NewtonCollisionPipelineCfg, SDFCfg
```

Then add this field to the `NewtonCfg` class, after the existing `collision_cfg` field:

```python
    sdf_cfg: SDFCfg | None = None
    """SDF collision configuration.

    When set, mesh collision shapes matching the configured patterns will have
    SDF built via Newton's ``mesh.build_sdf()`` at simulation start.  This
    also forces Newton's collision pipeline to be active (overriding
    ``use_mujoco_contacts=True`` if necessary).

    See :class:`~isaaclab_newton.physics.newton_collision_cfg.SDFCfg` for
    available parameters.
    """
```

- [ ] **Step 2: Update `__init__.pyi`**

Add `SDFCfg` to `__all__` and the import from `newton_collision_cfg`:

```python
__all__ = [
    "FeatherstoneSolverCfg",
    "HydroelasticSDFCfg",
    "MJWarpSolverCfg",
    "NewtonCfg",
    "NewtonCollisionPipelineCfg",
    "NewtonManager",
    "NewtonSolverCfg",
    "SDFCfg",
    "XPBDSolverCfg",
]

from .newton_collision_cfg import HydroelasticSDFCfg, NewtonCollisionPipelineCfg, SDFCfg
from .newton_manager import NewtonManager
from .newton_manager_cfg import (
    FeatherstoneSolverCfg,
    MJWarpSolverCfg,
    NewtonCfg,
    NewtonSolverCfg,
    XPBDSolverCfg,
)
```

- [ ] **Step 3: Run pre-commit**

```bash
./isaaclab.sh -f
```

- [ ] **Step 4: Commit**

```bash
git add source/isaaclab_newton/isaaclab_newton/physics/newton_manager_cfg.py source/isaaclab_newton/isaaclab_newton/physics/__init__.pyi
git commit -m "Wire SDFCfg into NewtonCfg and update exports"
```

---

### Task 3: Add SDF manager methods

**Files:**
- Modify: `source/isaaclab_newton/isaaclab_newton/physics/newton_manager.py`

- [ ] **Step 1: Add `re` import**

Add `import re` to the imports at the top of `newton_manager.py` (after `import logging`).

- [ ] **Step 2: Add `_build_sdf_on_mesh` static method**

Add this method to `NewtonManager`, after the `add_model_change` method (around line 392):

```python
    @staticmethod
    def _build_sdf_on_mesh(mesh, sdf_cfg, res_overrides, label: str):
        """Build SDF on a mesh, resolving per-pattern resolution overrides.

        Args:
            mesh: Newton mesh object to build SDF on.
            sdf_cfg: The active :class:`SDFCfg` instance.
            res_overrides: Compiled ``(pattern, resolution)`` pairs, or ``None``.
            label: Shape label used for pattern resolution matching.
        """
        if mesh is None:
            return
        if mesh.sdf is not None:
            mesh.clear_sdf()
        resolution = sdf_cfg.max_resolution
        if res_overrides is not None:
            for pat, res in res_overrides:
                if pat.search(label):
                    resolution = res
                    break
        sdf_kwargs: dict = dict(narrow_band_range=sdf_cfg.narrow_band_range)
        if resolution is not None:
            sdf_kwargs["max_resolution"] = resolution
        if sdf_cfg.target_voxel_size is not None:
            sdf_kwargs["target_voxel_size"] = sdf_cfg.target_voxel_size
        mesh.build_sdf(**sdf_kwargs)
```

- [ ] **Step 3: Add `_create_sdf_collision_from_visual` classmethod**

Add this method right after `_build_sdf_on_mesh`:

```python
    @classmethod
    def _create_sdf_collision_from_visual(
        cls, builder: ModelBuilder, sdf_shape_indices: set[int], sdf_cfg, res_overrides
    ):
        """Create collision shapes from visual meshes for matched bodies lacking collision geometry.

        Args:
            builder: Newton model builder to modify.
            sdf_shape_indices: Shape indices that matched SDF patterns.
            sdf_cfg: The active :class:`SDFCfg` instance.
            res_overrides: Compiled ``(pattern, resolution)`` pairs, or ``None``.

        Returns:
            Tuple of ``(num_added, num_hydro)`` counts.
        """
        from newton import ShapeFlags

        matched_bodies: set[int] = {builder.shape_body[si] for si in sdf_shape_indices}
        bodies_with_collision: set[int] = set()
        for si in range(builder.shape_count):
            if builder.shape_flags[si] & ShapeFlags.COLLIDE_SHAPES and builder.shape_body[si] in matched_bodies:
                bodies_with_collision.add(builder.shape_body[si])

        shape_cfg_kwargs: dict = dict(
            density=0.0,
            has_shape_collision=True,
            has_particle_collision=True,
            is_visible=False,
        )
        if sdf_cfg.margin is not None:
            shape_cfg_kwargs["margin"] = sdf_cfg.margin
        if sdf_cfg.k_hydro is not None:
            shape_cfg_kwargs["is_hydroelastic"] = True
            shape_cfg_kwargs["kh"] = sdf_cfg.k_hydro
        sdf_shape_cfg = ModelBuilder.ShapeConfig(**shape_cfg_kwargs)

        num_added = 0
        num_hydro = 0
        for body_idx in matched_bodies - bodies_with_collision:
            visual_si = None
            for si in sdf_shape_indices:
                if builder.shape_body[si] == body_idx and builder.shape_source[si] is not None:
                    visual_si = si
                    break
            if visual_si is None:
                body_lbl = builder.body_label[body_idx]
                logger.warning(f"SDF: body '{body_lbl}' matched but has no visual mesh to create collision from.")
                continue

            mesh = builder.shape_source[visual_si]
            cls._build_sdf_on_mesh(mesh, sdf_cfg, res_overrides, builder.shape_label[visual_si])

            body_lbl = builder.body_label[body_idx]
            builder.add_shape_mesh(
                body=body_idx,
                xform=builder.shape_transform[visual_si],
                mesh=mesh,
                scale=builder.shape_scale[visual_si],
                cfg=sdf_shape_cfg,
                label=f"{body_lbl}/sdf_collision",
            )
            num_added += 1
            if sdf_cfg.k_hydro is not None:
                num_hydro += 1

        return num_added, num_hydro
```

- [ ] **Step 4: Add `_apply_sdf_config` classmethod**

Add this method right after `_create_sdf_collision_from_visual`:

```python
    @classmethod
    def _apply_sdf_config(cls, builder: ModelBuilder):
        """Apply SDF collision and optional hydroelastic flags to matching mesh shapes.

        Reads :attr:`SDFCfg` from the active physics config. Collects shapes
        matching body/shape regex patterns, builds SDF on their meshes, and
        optionally sets the ``HYDROELASTIC`` flag with :attr:`SDFCfg.k_hydro`.

        Args:
            builder: Newton model builder to modify (before finalization).
        """
        from newton import GeoType, ShapeFlags

        cfg = PhysicsManager._cfg
        if cfg is None:
            return
        sdf_cfg = getattr(cfg, "sdf_cfg", None)
        if sdf_cfg is None:
            return

        if sdf_cfg.max_resolution is None and sdf_cfg.target_voxel_size is None:
            logger.warning("SDFCfg provided but neither max_resolution nor target_voxel_size is set. SDF disabled.")
            return

        # Compile patterns
        body_patterns = [re.compile(p) for p in sdf_cfg.body_patterns] if sdf_cfg.body_patterns else None
        shape_patterns = [re.compile(p) for p in sdf_cfg.shape_patterns] if sdf_cfg.shape_patterns else None
        res_overrides = (
            [(re.compile(p), r) for p, r in sdf_cfg.pattern_resolutions.items()]
            if sdf_cfg.pattern_resolutions
            else None
        )
        hydro_patterns = None
        if sdf_cfg.k_hydro is not None and sdf_cfg.hydroelastic_shape_patterns is not None:
            hydro_patterns = [re.compile(p) for p in sdf_cfg.hydroelastic_shape_patterns]

        if body_patterns is None and shape_patterns is None:
            logger.warning("SDFCfg has no body_patterns or shape_patterns set. No shapes will receive SDF.")
            return

        # Build reverse map: body_idx -> [mesh shape indices]
        body_to_shapes: dict[int, list[int]] = {}
        for si in range(builder.shape_count):
            if builder.shape_type[si] == GeoType.MESH:
                body_to_shapes.setdefault(builder.shape_body[si], []).append(si)

        sdf_shape_indices: set[int] = set()

        if body_patterns is not None:
            for body_idx in range(len(builder.body_label)):
                if any(p.search(builder.body_label[body_idx]) for p in body_patterns):
                    sdf_shape_indices.update(body_to_shapes.get(body_idx, []))

        if shape_patterns is not None:
            for shape_indices in body_to_shapes.values():
                for si in shape_indices:
                    if any(p.search(builder.shape_label[si]) for p in shape_patterns):
                        sdf_shape_indices.add(si)

        # Patch existing collision meshes
        num_patched = 0
        num_hydro = 0
        for si in sdf_shape_indices:
            if not (builder.shape_flags[si] & ShapeFlags.COLLIDE_SHAPES):
                continue
            cls._build_sdf_on_mesh(builder.shape_source[si], sdf_cfg, res_overrides, builder.shape_label[si])
            if sdf_cfg.margin is not None:
                builder.shape_margin[si] = sdf_cfg.margin
            if sdf_cfg.k_hydro is not None:
                apply_hydro = hydro_patterns is None or any(
                    p.search(builder.shape_label[si]) for p in hydro_patterns
                )
                if apply_hydro:
                    builder.shape_flags[si] |= ShapeFlags.HYDROELASTIC
                    builder.shape_material_kh[si] = sdf_cfg.k_hydro
                    num_hydro += 1
            num_patched += 1

        # Optionally create collision shapes from visual meshes
        num_added = 0
        if sdf_cfg.use_visual_meshes:
            num_added, hydro_from_visual = cls._create_sdf_collision_from_visual(
                builder, sdf_shape_indices, sdf_cfg, res_overrides
            )
            num_hydro += hydro_from_visual

        hydro_msg = f", {num_hydro} hydroelastic shape(s)" if sdf_cfg.k_hydro is not None else ""
        logger.info(
            f"SDF config: {num_added} collision shape(s) added, {num_patched} existing shape(s) patched{hydro_msg}. "
            f"(max_resolution={sdf_cfg.max_resolution}, narrow_band={sdf_cfg.narrow_band_range})"
        )
```

- [ ] **Step 5: Run pre-commit**

```bash
./isaaclab.sh -f
```

- [ ] **Step 6: Commit**

```bash
git add source/isaaclab_newton/isaaclab_newton/physics/newton_manager.py
git commit -m "Add SDF manager methods for shape preparation"
```

---

### Task 4: Integrate SDF into manager lifecycle

**Files:**
- Modify: `source/isaaclab_newton/isaaclab_newton/physics/newton_manager.py`

- [ ] **Step 1: Call `_apply_sdf_config` in `instantiate_builder_from_stage`**

In `instantiate_builder_from_stage()`, add a call to `_apply_sdf_config` just before `cls.set_builder(builder)` (around line 527):

```python
        cls._apply_sdf_config(builder)
        cls.set_builder(builder)
```

- [ ] **Step 2: Force collision pipeline in `initialize_solver` when SDF is configured**

In `initialize_solver()`, after the existing `if isinstance(cls._solver, SolverMuJoCo):` / `else:` block that sets `cls._needs_collision_pipeline` (after line 618), add:

```python
            # Force Newton pipeline when collision_cfg or SDF is configured
            if cfg.collision_cfg is not None and not cls._needs_collision_pipeline:
                logger.warning("collision_cfg set — enabling Newton collision pipeline.")
                cls._needs_collision_pipeline = True

            sdf_cfg = getattr(cfg, "sdf_cfg", None)
            has_sdf = (
                sdf_cfg is not None
                and (sdf_cfg.body_patterns is not None or sdf_cfg.shape_patterns is not None)
                and (sdf_cfg.max_resolution is not None or sdf_cfg.target_voxel_size is not None)
            )
            if has_sdf and not cls._needs_collision_pipeline:
                logger.warning("SDF collision requires Newton collision pipeline. Overriding use_mujoco_contacts.")
                cls._needs_collision_pipeline = True
```

- [ ] **Step 3: Add hydroelastic warning in `_initialize_contacts`**

In `_initialize_contacts()`, after the pipeline is created (after the `cls._collision_pipeline = CollisionPipeline(...)` lines, around line 542), add a warning check:

```python
                # Warn if hydroelastic was requested but no shapes qualify
                hydro_requested = (
                    cls._collision_cfg is not None
                    and cls._collision_cfg.sdf_hydroelastic_config is not None
                )
                if hydro_requested and cls._collision_pipeline.hydroelastic_sdf is None:
                    logger.warning(
                        "HydroelasticSDFCfg was set but no hydroelastic shape pairs found. "
                        "Ensure shapes have SDF built (via SDFCfg with k_hydro set) and that "
                        "both shapes in each contact pair have the HYDROELASTIC flag."
                    )
```

This goes after the `else: cls._collision_pipeline = CollisionPipeline(...)` branch but before the `if cls._contacts is None:` line, so it applies regardless of whether `_collision_cfg` was set or not. The check should be at the same indent level as the `if cls._collision_cfg is not None:` block:

```python
        if cls._needs_collision_pipeline:
            # Newton collision pipeline: create pipeline and generate contacts
            if cls._collision_pipeline is None:
                if cls._collision_cfg is not None:
                    cls._collision_pipeline = CollisionPipeline(cls._model, **cls._collision_cfg.to_pipeline_args())
                else:
                    cls._collision_pipeline = CollisionPipeline(cls._model, broad_phase="explicit")

                # Warn if hydroelastic was requested but no shapes qualify
                hydro_requested = (
                    cls._collision_cfg is not None
                    and cls._collision_cfg.sdf_hydroelastic_config is not None
                )
                if hydro_requested and cls._collision_pipeline.hydroelastic_sdf is None:
                    logger.warning(
                        "HydroelasticSDFCfg was set but no hydroelastic shape pairs found. "
                        "Ensure shapes have SDF built (via SDFCfg with k_hydro set) and that "
                        "both shapes in each contact pair have the HYDROELASTIC flag."
                    )

            if cls._contacts is None:
                cls._contacts = cls._collision_pipeline.contacts()
```

- [ ] **Step 4: Run pre-commit**

```bash
./isaaclab.sh -f
```

- [ ] **Step 5: Commit**

```bash
git add source/isaaclab_newton/isaaclab_newton/physics/newton_manager.py
git commit -m "Integrate SDF config into manager lifecycle"
```

---

### Task 5: Cloner integration

**Files:**
- Modify: `source/isaaclab_newton/isaaclab_newton/cloner/newton_replicate.py`

- [ ] **Step 1: Add imports**

Add `re` and `GeoType` imports at the top of `newton_replicate.py`:

```python
import re
```

(after `from __future__ import annotations`)

And update the newton import:

```python
from newton import GeoType, ModelBuilder, solvers
```

And add the `PhysicsManager` import:

```python
from isaaclab.physics import PhysicsManager
```

- [ ] **Step 2: Add SDF pattern skip and prototype SDF application**

In `_build_newton_builder_from_mapping`, after the `env0_pos = positions[0]` line (line 64), add SDF pattern compilation:

```python
    # SDF collision requires original triangle meshes for mesh.build_sdf().
    # Convex hull approximation destroys the source geometry, so shapes
    # matching SDF patterns must be excluded from approximation here.
    # _apply_sdf_config() builds the SDF on each prototype after approximation.
    cfg = PhysicsManager._cfg
    sdf_cfg = getattr(cfg, "sdf_cfg", None) if cfg is not None else None
    body_pats = [re.compile(x) for x in sdf_cfg.body_patterns] if sdf_cfg and sdf_cfg.body_patterns else None
    shape_pats = [re.compile(x) for x in sdf_cfg.shape_patterns] if sdf_cfg and sdf_cfg.shape_patterns else None
    has_sdf_patterns = body_pats is not None or shape_pats is not None
```

Then replace the existing `if simplify_meshes:` block (line 76-77):

```python
        if simplify_meshes:
            p.approximate_meshes("convex_hull", keep_visual_shapes=True)
        protos[src_path] = p
```

with:

```python
        if simplify_meshes:
            if has_sdf_patterns:
                sdf_bodies: set[int] = set()
                if body_pats is not None:
                    for bi in range(len(p.body_label)):
                        if any(pat.search(p.body_label[bi]) for pat in body_pats):
                            sdf_bodies.add(bi)

                approx_indices = []
                for i in range(len(p.shape_type)):
                    if p.shape_type[i] != GeoType.MESH:
                        continue
                    # Skip shapes that will use SDF (matched by body or shape pattern)
                    if p.shape_body[i] in sdf_bodies:
                        continue
                    if shape_pats is not None:
                        lbl = p.shape_label[i] if i < len(p.shape_label) else ""
                        if any(pat.search(lbl) for pat in shape_pats):
                            continue
                    approx_indices.append(i)
                if approx_indices:
                    p.approximate_meshes("convex_hull", shape_indices=approx_indices, keep_visual_shapes=True)
            else:
                p.approximate_meshes("convex_hull", keep_visual_shapes=True)
        # Build SDF on prototype before add_builder copies it N times.
        # Mesh objects are shared by reference, so SDF is built once and
        # all environments inherit it.
        NewtonManager._apply_sdf_config(p)
        protos[src_path] = p
```

- [ ] **Step 3: Run pre-commit**

```bash
./isaaclab.sh -f
```

- [ ] **Step 4: Commit**

```bash
git add source/isaaclab_newton/isaaclab_newton/cloner/newton_replicate.py
git commit -m "Skip convex hull for SDF shapes and apply SDF on cloner prototypes"
```

---

### Task 6: Tests

**Files:**
- Create: `source/isaaclab_newton/test/physics/test_sdf_config.py`

- [ ] **Step 1: Create test file**

Create `source/isaaclab_newton/test/physics/test_sdf_config.py` with the following content. Tests use `unittest.mock` to avoid needing a running Newton simulation:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for SDF collision configuration and application logic."""

import re
from unittest.mock import MagicMock, patch

from newton import GeoType, ModelBuilder, ShapeFlags


class TestBuildSdfOnMesh:
    """Tests for NewtonManager._build_sdf_on_mesh."""

    @staticmethod
    def _make_sdf_cfg(max_resolution=256, narrow_band_range=(-0.1, 0.1), target_voxel_size=None):
        cfg = MagicMock()
        cfg.max_resolution = max_resolution
        cfg.narrow_band_range = narrow_band_range
        cfg.target_voxel_size = target_voxel_size
        return cfg

    def test_none_mesh_is_noop(self):
        """Passing None as mesh should not raise."""
        from isaaclab_newton.physics.newton_manager import NewtonManager

        sdf_cfg = self._make_sdf_cfg()
        NewtonManager._build_sdf_on_mesh(None, sdf_cfg, None, "test_label")

    def test_builds_sdf_with_max_resolution(self):
        """SDF is built on mesh with max_resolution and narrow_band_range."""
        from isaaclab_newton.physics.newton_manager import NewtonManager

        mesh = MagicMock()
        mesh.sdf = None
        sdf_cfg = self._make_sdf_cfg(max_resolution=128)

        NewtonManager._build_sdf_on_mesh(mesh, sdf_cfg, None, "test_label")

        mesh.build_sdf.assert_called_once_with(narrow_band_range=(-0.1, 0.1), max_resolution=128)

    def test_clears_existing_sdf_before_rebuild(self):
        """Existing SDF on mesh is cleared before building a new one."""
        from isaaclab_newton.physics.newton_manager import NewtonManager

        mesh = MagicMock()
        mesh.sdf = "existing_sdf"
        sdf_cfg = self._make_sdf_cfg()

        NewtonManager._build_sdf_on_mesh(mesh, sdf_cfg, None, "test_label")

        mesh.clear_sdf.assert_called_once()
        mesh.build_sdf.assert_called_once()

    def test_target_voxel_size_passed_alongside_resolution(self):
        """When target_voxel_size is set, it is passed alongside max_resolution."""
        from isaaclab_newton.physics.newton_manager import NewtonManager

        mesh = MagicMock()
        mesh.sdf = None
        sdf_cfg = self._make_sdf_cfg(max_resolution=256, target_voxel_size=0.005)

        NewtonManager._build_sdf_on_mesh(mesh, sdf_cfg, None, "test_label")

        call_kwargs = mesh.build_sdf.call_args[1]
        assert call_kwargs["target_voxel_size"] == 0.005
        assert call_kwargs["max_resolution"] == 256

    def test_resolution_override_by_pattern(self):
        """Per-pattern resolution override is applied when label matches."""
        from isaaclab_newton.physics.newton_manager import NewtonManager

        mesh = MagicMock()
        mesh.sdf = None
        sdf_cfg = self._make_sdf_cfg(max_resolution=256)
        res_overrides = [(re.compile(".*elbow.*"), 128)]

        NewtonManager._build_sdf_on_mesh(mesh, sdf_cfg, res_overrides, "/World/Robot/elbow_link/collision")

        call_kwargs = mesh.build_sdf.call_args[1]
        assert call_kwargs["max_resolution"] == 128

    def test_resolution_override_no_match_uses_global(self):
        """When label doesn't match any override, global max_resolution is used."""
        from isaaclab_newton.physics.newton_manager import NewtonManager

        mesh = MagicMock()
        mesh.sdf = None
        sdf_cfg = self._make_sdf_cfg(max_resolution=256)
        res_overrides = [(re.compile(".*elbow.*"), 128)]

        NewtonManager._build_sdf_on_mesh(mesh, sdf_cfg, res_overrides, "/World/Robot/wrist_link/collision")

        call_kwargs = mesh.build_sdf.call_args[1]
        assert call_kwargs["max_resolution"] == 256

    def test_resolution_override_first_match_wins(self):
        """First matching pattern in res_overrides determines resolution."""
        from isaaclab_newton.physics.newton_manager import NewtonManager

        mesh = MagicMock()
        mesh.sdf = None
        sdf_cfg = self._make_sdf_cfg(max_resolution=256)
        res_overrides = [
            (re.compile(".*link.*"), 64),
            (re.compile(".*elbow.*"), 128),
        ]

        NewtonManager._build_sdf_on_mesh(mesh, sdf_cfg, res_overrides, "/World/Robot/elbow_link/collision")

        call_kwargs = mesh.build_sdf.call_args[1]
        assert call_kwargs["max_resolution"] == 64  # ".*link.*" matches first


class TestApplySdfConfig:
    """Tests for NewtonManager._apply_sdf_config shape index collection and patching."""

    @staticmethod
    def _make_builder(bodies, shapes):
        """Create a minimal ModelBuilder-like mock.

        Args:
            bodies: List of body label strings.
            shapes: List of dicts with keys: body_idx, label, geo_type, flags, source.
        """
        builder = MagicMock(spec=ModelBuilder)
        builder.body_label = bodies
        builder.shape_count = len(shapes)
        builder.shape_type = [s["geo_type"] for s in shapes]
        builder.shape_body = [s["body_idx"] for s in shapes]
        builder.shape_label = [s["label"] for s in shapes]
        builder.shape_flags = [s["flags"] for s in shapes]
        builder.shape_source = [s.get("source") for s in shapes]
        builder.shape_margin = [0.0] * len(shapes)
        builder.shape_material_kh = [0.0] * len(shapes)
        return builder

    @staticmethod
    def _make_cfg(
        body_patterns=None,
        shape_patterns=None,
        max_resolution=256,
        k_hydro=None,
        hydroelastic_shape_patterns=None,
    ):
        cfg = MagicMock()
        cfg.sdf_cfg = MagicMock()
        cfg.sdf_cfg.max_resolution = max_resolution
        cfg.sdf_cfg.target_voxel_size = None
        cfg.sdf_cfg.narrow_band_range = (-0.1, 0.1)
        cfg.sdf_cfg.margin = None
        cfg.sdf_cfg.body_patterns = body_patterns
        cfg.sdf_cfg.shape_patterns = shape_patterns
        cfg.sdf_cfg.pattern_resolutions = None
        cfg.sdf_cfg.use_visual_meshes = False
        cfg.sdf_cfg.k_hydro = k_hydro
        cfg.sdf_cfg.hydroelastic_shape_patterns = hydroelastic_shape_patterns
        return cfg

    def test_no_sdf_cfg_is_noop(self):
        """_apply_sdf_config returns early when sdf_cfg is None."""
        from isaaclab_newton.physics.newton_manager import NewtonManager

        builder = MagicMock(spec=ModelBuilder)
        with patch("isaaclab_newton.physics.newton_manager.PhysicsManager") as pm:
            pm._cfg = MagicMock()
            pm._cfg.sdf_cfg = None
            NewtonManager._apply_sdf_config(builder)
        # No crash, no calls
        assert not builder.method_calls

    def test_no_patterns_warns(self):
        """_apply_sdf_config warns when no patterns are set."""
        from isaaclab_newton.physics.newton_manager import NewtonManager

        builder = MagicMock(spec=ModelBuilder)
        cfg = self._make_cfg(body_patterns=None, shape_patterns=None)
        with patch("isaaclab_newton.physics.newton_manager.PhysicsManager") as pm:
            pm._cfg = cfg
            with patch("isaaclab_newton.physics.newton_manager.logger") as mock_logger:
                NewtonManager._apply_sdf_config(builder)
                mock_logger.warning.assert_called()

    def test_body_pattern_collects_shapes(self):
        """Shapes under matching bodies are collected for SDF."""
        from isaaclab_newton.physics.newton_manager import NewtonManager

        bodies = ["/World/Robot/elbow", "/World/Robot/wrist"]
        shapes = [
            {"body_idx": 0, "label": "/World/Robot/elbow/col", "geo_type": GeoType.MESH,
             "flags": ShapeFlags.COLLIDE_SHAPES, "source": MagicMock(sdf=None)},
            {"body_idx": 1, "label": "/World/Robot/wrist/col", "geo_type": GeoType.MESH,
             "flags": ShapeFlags.COLLIDE_SHAPES, "source": MagicMock(sdf=None)},
        ]
        builder = self._make_builder(bodies, shapes)
        cfg = self._make_cfg(body_patterns=[".*elbow.*"])

        with patch("isaaclab_newton.physics.newton_manager.PhysicsManager") as pm:
            pm._cfg = cfg
            NewtonManager._apply_sdf_config(builder)

        # Only elbow shape should have build_sdf called
        shapes[0]["source"].build_sdf.assert_called_once()
        shapes[1]["source"].build_sdf.assert_not_called()

    def test_hydroelastic_flag_set_when_k_hydro(self):
        """HYDROELASTIC flag is set on matched shapes when k_hydro is provided."""
        from isaaclab_newton.physics.newton_manager import NewtonManager

        bodies = ["/World/Robot/elbow"]
        shapes = [
            {"body_idx": 0, "label": "/World/Robot/elbow/col", "geo_type": GeoType.MESH,
             "flags": ShapeFlags.COLLIDE_SHAPES, "source": MagicMock(sdf=None)},
        ]
        builder = self._make_builder(bodies, shapes)
        cfg = self._make_cfg(body_patterns=[".*elbow.*"], k_hydro=1e10)

        with patch("isaaclab_newton.physics.newton_manager.PhysicsManager") as pm:
            pm._cfg = cfg
            NewtonManager._apply_sdf_config(builder)

        assert builder.shape_flags[0] & ShapeFlags.HYDROELASTIC
        assert builder.shape_material_kh[0] == 1e10

    def test_hydroelastic_shape_patterns_filter(self):
        """hydroelastic_shape_patterns limits which shapes get HYDROELASTIC flag."""
        from isaaclab_newton.physics.newton_manager import NewtonManager

        bodies = ["/World/Robot/elbow", "/World/Robot/wrist"]
        shapes = [
            {"body_idx": 0, "label": "/World/Robot/elbow/col", "geo_type": GeoType.MESH,
             "flags": ShapeFlags.COLLIDE_SHAPES, "source": MagicMock(sdf=None)},
            {"body_idx": 1, "label": "/World/Robot/wrist/col", "geo_type": GeoType.MESH,
             "flags": ShapeFlags.COLLIDE_SHAPES, "source": MagicMock(sdf=None)},
        ]
        builder = self._make_builder(bodies, shapes)
        cfg = self._make_cfg(
            body_patterns=[".*"],
            k_hydro=1e10,
            hydroelastic_shape_patterns=[".*elbow.*"],
        )

        with patch("isaaclab_newton.physics.newton_manager.PhysicsManager") as pm:
            pm._cfg = cfg
            NewtonManager._apply_sdf_config(builder)

        # Both get SDF built
        shapes[0]["source"].build_sdf.assert_called_once()
        shapes[1]["source"].build_sdf.assert_called_once()
        # Only elbow gets hydroelastic
        assert builder.shape_flags[0] & ShapeFlags.HYDROELASTIC
        assert not (builder.shape_flags[1] & ShapeFlags.HYDROELASTIC)

    def test_shape_pattern_matching(self):
        """shape_patterns directly matches shape labels."""
        from isaaclab_newton.physics.newton_manager import NewtonManager

        bodies = ["/World/Robot/body"]
        shapes = [
            {"body_idx": 0, "label": "/World/Robot/body/Gear_col", "geo_type": GeoType.MESH,
             "flags": ShapeFlags.COLLIDE_SHAPES, "source": MagicMock(sdf=None)},
            {"body_idx": 0, "label": "/World/Robot/body/frame_col", "geo_type": GeoType.MESH,
             "flags": ShapeFlags.COLLIDE_SHAPES, "source": MagicMock(sdf=None)},
        ]
        builder = self._make_builder(bodies, shapes)
        cfg = self._make_cfg(shape_patterns=[".*Gear.*"])

        with patch("isaaclab_newton.physics.newton_manager.PhysicsManager") as pm:
            pm._cfg = cfg
            NewtonManager._apply_sdf_config(builder)

        shapes[0]["source"].build_sdf.assert_called_once()
        shapes[1]["source"].build_sdf.assert_not_called()

    def test_non_mesh_shapes_skipped(self):
        """Non-mesh shapes are never collected for SDF."""
        from isaaclab_newton.physics.newton_manager import NewtonManager

        bodies = ["/World/Robot/elbow"]
        shapes = [
            {"body_idx": 0, "label": "/World/Robot/elbow/box", "geo_type": GeoType.BOX,
             "flags": ShapeFlags.COLLIDE_SHAPES, "source": None},
        ]
        builder = self._make_builder(bodies, shapes)
        cfg = self._make_cfg(body_patterns=[".*elbow.*"])

        with patch("isaaclab_newton.physics.newton_manager.PhysicsManager") as pm:
            pm._cfg = cfg
            NewtonManager._apply_sdf_config(builder)
        # No build_sdf calls (box shape has no source to call on)
```

- [ ] **Step 2: Run tests**

```bash
./isaaclab.sh -p -m pytest source/isaaclab_newton/test/physics/test_sdf_config.py -v
```

Expected: All tests pass.

- [ ] **Step 3: Commit**

```bash
git add source/isaaclab_newton/test/physics/test_sdf_config.py
git commit -m "Add tests for SDF config and shape preparation"
```

---

### Task 7: Changelog and version bump

**Files:**
- Modify: `source/isaaclab_newton/docs/CHANGELOG.rst`
- Modify: `source/isaaclab_newton/config/extension.toml`

- [ ] **Step 1: Bump version to 0.5.12**

In `source/isaaclab_newton/config/extension.toml`, change:
```
version = "0.5.11"
```
to:
```
version = "0.5.12"
```

- [ ] **Step 2: Add new changelog version**

In `source/isaaclab_newton/docs/CHANGELOG.rst`, add a new version heading before `0.5.11`:

```rst
0.5.12 (2026-04-10)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :class:`~isaaclab_newton.physics.SDFCfg` for configuring SDF-based mesh
  collisions via Newton's ``mesh.build_sdf()`` API. Supports per-body and per-shape
  regex pattern matching, per-pattern resolution overrides, and optional creation of
  collision shapes from visual meshes.
* Added hydroelastic shape enablement fields
  (:attr:`~isaaclab_newton.physics.SDFCfg.k_hydro`,
  :attr:`~isaaclab_newton.physics.SDFCfg.hydroelastic_shape_patterns`) on
  :class:`~isaaclab_newton.physics.SDFCfg`.
* Added missing hydroelastic pipeline parameters to
  :class:`~isaaclab_newton.physics.HydroelasticSDFCfg`: ``moment_matching``,
  ``buffer_mult_broad``, ``buffer_mult_iso``, ``buffer_mult_contact``, ``grid_size``.
* Added SDF pattern skip in the Newton cloner to preserve original triangle
  meshes for shapes that will use SDF collision.


```

- [ ] **Step 3: Run pre-commit**

```bash
./isaaclab.sh -f
```

- [ ] **Step 4: Commit**

```bash
git add source/isaaclab_newton/docs/CHANGELOG.rst source/isaaclab_newton/config/extension.toml
git commit -m "Add SDF changelog entries and bump to 0.5.12"
```

---

### Task 8: Final validation

- [ ] **Step 1: Run all tests**

```bash
./isaaclab.sh -p -m pytest source/isaaclab_newton/test/physics/test_sdf_config.py -v
```

Expected: All tests pass.

- [ ] **Step 2: Run pre-commit on all files**

```bash
./isaaclab.sh -f
```

Expected: All checks pass.

- [ ] **Step 3: Verify diff is clean**

```bash
git diff
git status
```

Expected: No unstaged changes.
