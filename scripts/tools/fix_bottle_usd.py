"""Fix bottle.usd exported from URDF so it can be loaded as a RigidObject in Isaac Lab.

URDF exports contain IsaacRobotAPI, IsaacLinkAPI, and PhysicsFixedJoint which
conflict with RigidObject loading and cause PhysX simulation view invalidation.

Run inside the Isaac Sim container:
    isaaclab --python scripts/tools/fix_bottle_usd.py --headless
"""

# Launch Isaac Sim Simulator first (required for pxr module availability).
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Fix bottle.usd for RigidObject usage.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

"""Rest everything follows."""

import os
from pxr import Usd, UsdGeom, UsdPhysics

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "../.."))
USD_PATH = os.path.join(REPO_ROOT, "usd/bottle/bottle.usd")
FIXED_MARKER = os.path.join(REPO_ROOT, "usd/bottle/.fix_applied")

# Check if already fixed
if os.path.exists(FIXED_MARKER):
    print("bottle.usd already fixed — skipping. Delete usd/bottle/.fix_applied to force re-fix.")
    simulation_app.close()
    exit(0)

print(f"Opening: {USD_PATH}")
stage = Usd.Stage.Open(USD_PATH)
count = 0

# Remove IsaacRobotAPI and IsaacLinkAPI (leftover from URDF export)
for prim in stage.TraverseAll():
    applied = prim.GetAppliedSchemas()
    if "IsaacRobotAPI" in applied:
        prim.RemoveAPI("IsaacRobotAPI")
        count += 1
        print(f"  Removed IsaacRobotAPI from: {prim.GetPath()}")
    if "IsaacLinkAPI" in applied:
        prim.RemoveAPI("IsaacLinkAPI")
        count += 1
        print(f"  Removed IsaacLinkAPI from: {prim.GetPath()}")

# Deactivate PhysicsFixedJoint prims
for prim in stage.TraverseAll():
    if prim.IsA(UsdPhysics.FixedJoint):
        prim.SetActive(False)
        count += 1
        print(f"  Deactivated FixedJoint: {prim.GetPath()}")

# Ensure mesh prims have PhysicsRigidBodyAPI and CollisionAPI
for prim in stage.TraverseAll():
    if prim.IsA(UsdGeom.Mesh):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI.Apply(prim)
            count += 1
            print(f"  Added RigidBodyAPI to: {prim.GetPath()}")
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(prim)
            count += 1
            print(f"  Added CollisionAPI to: {prim.GetPath()}")

if count > 0:
    stage.GetRootLayer().Save()
    # Write marker file so we don't re-fix
    with open(FIXED_MARKER, "w") as f:
        f.write("fixed\n")
    print(f"\nDone. Cleaned {count} item(s), saved to {USD_PATH}")
else:
    print("No changes needed.")

simulation_app.close()
