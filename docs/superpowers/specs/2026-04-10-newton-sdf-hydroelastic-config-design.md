# Newton SDF & Hydroelastic Configuration

**Date:** 2026-04-10
**Built on:** PR #5219 (`expose_newton_collision_pipeline`)
**Ports from:** PR #5160 (`vidur/feature/sdf-collision`)
**Target:** New PR → `develop`

## Summary

Port SDF collision and hydroelastic shape preparation from PR #5160 into
PR #5219's config design. #5219 owns the collision pipeline config layer;
this PR adds the shape-preparation machinery that makes hydroelastic
contacts actually work end-to-end.

## Design Decisions

**Keep #5219's config hierarchy.** `HydroelasticSDFCfg` (pipeline processing
params) stays nested under `NewtonCollisionPipelineCfg.sdf_hydroelastic_config`.

**Add `SDFCfg` as a new top-level field on `NewtonCfg`.** Shape-level concerns
(which meshes get SDF, resolution, hydroelastic flag enablement) live here,
separate from pipeline configuration.

**Flatten hydroelastic shape enablement into `SDFCfg`.** #5160 used a nested
`HydroelasticCfg` object. We put `k_hydro` and `hydroelastic_shape_patterns`
directly on `SDFCfg` since they're simple fields that control shape flags,
not a separate subsystem.

## Config Hierarchy

```
NewtonCfg
├── solver_cfg: NewtonSolverCfg
├── collision_cfg: NewtonCollisionPipelineCfg | None
│   ├── broad_phase, reduce_contacts, rigid_contact_max, ...
│   └── sdf_hydroelastic_config: HydroelasticSDFCfg | None
│       ├── reduce_contacts, normal_matching, anchor_contact
│       ├── moment_matching, margin_contact_area
│       ├── buffer_fraction, buffer_mult_broad/iso/contact
│       ├── grid_size, output_contact_surface
│       └── (maps 1:1 to HydroelasticSDF.Config via to_pipeline_args())
└── sdf_cfg: SDFCfg | None
    ├── max_resolution: int | None
    ├── target_voxel_size: float | None
    ├── narrow_band_range: tuple[float, float]
    ├── margin: float | None
    ├── body_patterns: list[str] | None
    ├── shape_patterns: list[str] | None
    ├── pattern_resolutions: dict[str, int] | None
    ├── use_visual_meshes: bool
    ├── k_hydro: float              (shape-level stiffness)
    └── hydroelastic_shape_patterns: list[str] | None
```

## Components

### 1. Config additions (`newton_collision_cfg.py`)

**`SDFCfg`** — new configclass in the existing file.

Fields ported from #5160's `SDFCfg`:
- `max_resolution`, `target_voxel_size`, `narrow_band_range`, `margin`
- `body_patterns`, `shape_patterns`, `pattern_resolutions`
- `use_visual_meshes`

Shape-level hydroelastic fields (flattened from #5160's `HydroelasticCfg`):
- `k_hydro: float = 1e10` — stiffness applied to shapes via `shape_material_kh`
- `hydroelastic_shape_patterns: list[str] | None = None` — if None, all
  SDF shapes get HYDROELASTIC flag; if set, only matching shapes

**`HydroelasticSDFCfg`** — add missing pipeline params from #5160:
- `moment_matching: bool = False`
- `buffer_mult_broad: int = 1`
- `buffer_mult_iso: int = 1`
- `buffer_mult_contact: int = 1`
- `grid_size: int = 256 * 8 * 128`

**`NewtonCfg`** — add `sdf_cfg: SDFCfg | None = None` in `newton_manager_cfg.py`.

### 2. Manager methods (`newton_manager.py`)

**`_build_sdf_on_mesh(mesh, sdf_cfg, res_overrides, label)`** — static method.
Builds SDF on a mesh. Clears existing SDF first. Applies per-pattern
resolution overrides. Passes `narrow_band_range`, `max_resolution`,
`target_voxel_size` to `mesh.build_sdf()`.

**`_apply_sdf_config(builder)`** — classmethod.
1. Read `sdf_cfg` from `PhysicsManager._cfg`; return early if None.
2. Validate that at least one of `max_resolution`/`target_voxel_size` is set.
3. Compile body/shape/hydroelastic regex patterns.
4. Collect matching shape indices from builder.
5. For each matching collision shape: build SDF, optionally set HYDROELASTIC
   flag + `k_hydro`.
6. If `use_visual_meshes`, call `_create_sdf_collision_from_visual()`.
7. Log summary.

**`_create_sdf_collision_from_visual(builder, sdf_shape_indices, sdf_cfg, res_overrides)`**
— classmethod. For matched bodies that lack collision geometry, creates a
collision shape from the first visual mesh with SDF built on it.

**`initialize_solver()` changes:**
- After determining `_needs_collision_pipeline`, force it `True` when
  `collision_cfg is not None` or when `sdf_cfg` has valid patterns + resolution.
- Log a warning when overriding.

**`_initialize_contacts()` changes:**
- After creating pipeline, if hydroelastic was configured
  (`collision_cfg.sdf_hydroelastic_config is not None`) but
  `pipeline.hydroelastic_sdf is None`, log a warning.

**`instantiate_builder_from_stage()` change:**
- Call `cls._apply_sdf_config(builder)` before `cls.set_builder(builder)`.

### 3. Cloner integration (`newton_replicate.py`)

In `_build_newton_builder_from_mapping()`:
1. Read `sdf_cfg` from `PhysicsManager._cfg`.
2. Compile body/shape patterns if present.
3. When `simplify_meshes` is True and SDF patterns exist, skip convex hull
   approximation for shapes matching SDF patterns (preserves triangle meshes).
4. After prototype building, call `NewtonManager._apply_sdf_config(prototype)`
   on each prototype before `add_builder` replication.

### 4. Exports (`__init__.pyi`)

Add `SDFCfg` to `__all__` and import list.

### 5. Tests (`test_sdf_config.py`)

Port from #5160, adapted for new config structure:
- `TestBuildSdfOnMesh` — None mesh, max_resolution, clear existing SDF,
  target_voxel_size precedence, pattern resolution overrides
- `TestApplySdfConfig` — shape index collection by body/shape patterns,
  hydroelastic flag setting, visual mesh fallback, edge cases

### 6. Changelog

Add to existing `0.5.11` entry (or bump to `0.5.12`):
- Added `SDFCfg` for SDF mesh collision configuration
- Added SDF pattern skip in Newton cloner
- Added missing hydroelastic pipeline params to `HydroelasticSDFCfg`

## Execution Order

1. Config additions (SDFCfg, HydroelasticSDFCfg fields, NewtonCfg.sdf_cfg)
2. Manager methods (_build_sdf_on_mesh, _apply_sdf_config, _create_sdf_collision_from_visual)
3. Manager integration (initialize_solver, _initialize_contacts, instantiate_builder_from_stage)
4. Cloner integration (newton_replicate.py)
5. Exports (__init__.pyi)
6. Tests
7. Changelog + version bump

## Out of Scope

- Visualizer additions from #5160 (newton_visualizer.py changes)
- Runtime SDF rebuild / dynamic pattern updates
