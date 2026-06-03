"""Generic fix for URDF-exported USD files so they can be loaded as RigidObjects in Isaac Lab.

URDF exports contain IsaacRobotAPI, IsaacLinkAPI, PhysicsArticulationRootAPI,
PhysxArticulationAPI, and PhysicsFixedJoint which conflict with RigidObject
loading and cause PhysX simulation view invalidation.

This script opens the full USD composition (root layer + all sublayers),
traverses ALL prims across ALL layers, and removes the problematic APIs
from whichever layer defines them.

Usage (inside container):
    isaaclab --python scripts/tools/fix_usd_rigid_body.py --headless usd/bottle/bottle.usd
    isaaclab --python scripts/tools/fix_usd_rigid_body.py --headless usd/box/box.usd
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Fix URDF-exported USD for RigidObject usage.")
parser.add_argument("usd_path", type=str, help="Path to the root USD file (e.g., usd/bottle/bottle.usd)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

"""Rest everything follows."""

from pxr import Sdf, Usd

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "../.."))
full_path = os.path.join(REPO_ROOT, args_cli.usd_path) if not os.path.isabs(args_cli.usd_path) else args_cli.usd_path
full_path = os.path.normpath(full_path)

if not os.path.exists(full_path):
    print(f"ERROR: File not found: {full_path}")
    simulation_app.close()
    sys.exit(1)

BAD_APIS = {"IsaacRobotAPI", "IsaacLinkAPI", "PhysicsArticulationRootAPI", "PhysxArticulationAPI"}
BAD_PRIM_TYPE_TOKENS = {"PhysicsFixedJoint"}


def _iter_api_schemas(meta_val) -> list[str]:
    """Extract apiSchemas as a plain list, handling both VtArray and TokenListOp."""
    if meta_val is None:
        return []
    # TokenListOp (from Sdf layer editing)
    if hasattr(meta_val, "GetItems") and callable(meta_val.GetItems):
        return list(meta_val.GetItems())
    # Standard VtArray or list
    try:
        return list(meta_val)
    except TypeError:
        return []


def _update_api_schemas(prim_spec, remove_apis: set[str]):
    """Remove bad APIs from a prim spec's apiSchemas. Handles TokenListOp correctly."""
    meta_val = prim_spec.GetInfo("apiSchemas")
    if meta_val is None:
        return False

    # For TokenListOp, we need to preserve the list-op structure
    if hasattr(meta_val, "GetItems") and callable(meta_val.GetItems):
        # Get the resolved items
        resolved = list(meta_val.GetItems())
        new_resolved = [s for s in resolved if s not in remove_apis]
        if len(new_resolved) != len(resolved):
            # Rebuild as explicit list (not list op)
            if new_resolved:
                prim_spec.SetInfo("apiSchemas", new_resolved)
            else:
                prim_spec.ClearInfo("apiSchemas")
            return True
        return False

    # Plain list
    resolved = list(meta_val)
    new_resolved = [s for s in resolved if s not in remove_apis]
    if len(new_resolved) != len(resolved):
        if new_resolved:
            prim_spec.SetInfo("apiSchemas", new_resolved)
        else:
            prim_spec.ClearInfo("apiSchemas")
        return True
    return False


print(f"Opening root: {full_path}")

# Open the root stage (full composition) and fix all prims
# Use GetMetadata("apiSchemas") because custom schemas like IsaacRobotAPI
# are NOT returned by GetAppliedSchemas() when the schema plugin isn't loaded.
stage = Usd.Stage.Open(full_path)
if not stage:
    print(f"ERROR: Cannot open stage: {full_path}")
    simulation_app.close()
    sys.exit(1)

total_changes = 0
layer_changes: dict[str, list[str]] = {}

for prim in stage.TraverseAll():
    prim_path = str(prim.GetPath())

    # Check apiSchemas METADATA (not GetAppliedSchemas, which requires loaded schema plugins)
    api_schemas_meta = _iter_api_schemas(prim.GetMetadata("apiSchemas"))
    bad_on_prim = [s for s in api_schemas_meta if s in BAD_APIS]
    has_bad_type = prim.GetTypeName() in BAD_PRIM_TYPE_TOKENS

    if not bad_on_prim and not has_bad_type:
        continue

    # Find which layer defines this via prim stack
    prim_stack = prim.GetPrimStack()
    for prim_spec in prim_stack:
        spec_layer = prim_spec.layer
        if not spec_layer or not spec_layer.realPath:
            continue

        layer_key = spec_layer.realPath
        if layer_key not in layer_changes:
            layer_changes[layer_key] = []

        # Fix bad APIs in apiSchemas metadata of this prim spec
        spec_api_schemas = _iter_api_schemas(prim_spec.GetInfo("apiSchemas"))
        spec_bad = [s for s in spec_api_schemas if s in bad_on_prim and s in BAD_APIS]
        if spec_bad:
            if _update_api_schemas(prim_spec, set(spec_bad)):
                for api in spec_bad:
                    layer_changes[layer_key].append(f"Remove {api} from {prim_path}")

        # Fix bad prim types
        if has_bad_type and prim_spec.specifier == Sdf.SpecifierDef:
            layer_changes[layer_key].append(f"Deactivate {prim.GetTypeName()}: {prim_path}")
            prim.SetActive(False)

# Save all modified layers
for layer_path, edits in sorted(layer_changes.items()):
    if not edits:
        continue
    print(f"\nLayer: {layer_path}")
    for edit in edits:
        print(f"  {edit}")
        total_changes += 1
    layer = Sdf.Layer.FindOrOpen(layer_path)
    if layer:
        layer.Save()
        print(f"  -> Saved")

if total_changes > 0:
    marker_path = os.path.join(os.path.dirname(full_path), ".fix_applied_v3")
    with open(marker_path, "w") as f:
        f.write("fixed\n")
    print(f"\nDone: {total_changes} total changes")
    print(f"Marker: {marker_path}")
else:
    print("\nNo changes needed.")

simulation_app.close()
