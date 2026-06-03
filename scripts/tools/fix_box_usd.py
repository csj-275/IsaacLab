"""Fix box.usd exported from URDF so it can be loaded as a RigidObject in Isaac Lab.

URDF exports contain IsaacRobotAPI, IsaacLinkAPI, PhysicsArticulationRootAPI,
PhysxArticulationAPI, and PhysicsFixedJoint which conflict with RigidObject
loading and cause PhysX issues.

Both the root layer and the referenced urdf/box/box.usd are cleaned.

Run inside the Isaac Sim container:
    isaaclab --python scripts/tools/fix_box_usd.py --headless
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Fix box.usd for RigidObject usage.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

"""Rest everything follows."""

import os
from pxr import Sdf, Usd, UsdPhysics

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "../.."))
FIXED_MARKER = os.path.join(REPO_ROOT, "usd/box/.fix_applied")

if os.path.exists(FIXED_MARKER):
    os.remove(FIXED_MARKER)
    print("Removed old fix marker — will re-fix all layers including configuration sublayers.")

LAYER_PATHS = [
    os.path.join(REPO_ROOT, "usd/box/urdf/box/box.usd"),                            # referenced layer
    os.path.join(REPO_ROOT, "usd/box/box.usd"),                                      # root layer
    os.path.join(REPO_ROOT, "usd/box/urdf/box/configuration/box_robot.usd"),        # robot config (has IsaacRobotAPI)
    os.path.join(REPO_ROOT, "usd/box/urdf/box/configuration/box_physics.usd"),      # physics config (has ArticulationRoot)
]
BAD_APIS = {"IsaacRobotAPI", "IsaacLinkAPI", "PhysicsArticulationRootAPI", "PhysxArticulationAPI"}

total = 0
for layer_path in LAYER_PATHS:
    print(f"\nProcessing: {layer_path}")
    layer = Sdf.Layer.FindOrOpen(layer_path)
    if not layer:
        print(f"  SKIP: cannot open")
        continue

    stage = Usd.Stage.Open(layer.identifier)
    stage.SetEditTarget(layer)
    count = 0

    for prim in stage.TraverseAll():
        apis = list(prim.GetAppliedSchemas())
        for api in apis:
            if api in BAD_APIS:
                prim.RemoveAPI(api)
                count += 1
                print(f"  Removed {api} from {prim.GetPath()}")
        if prim.IsA(UsdPhysics.FixedJoint):
            prim.SetActive(False)
            count += 1
            print(f"  Deactivated FixedJoint: {prim.GetPath()}")

    if count > 0:
        layer.Save()
        print(f"  -> Saved {count} change(s)")
    total += count

if total > 0:
    with open(FIXED_MARKER, "w") as f:
        f.write("fixed\n")
    print(f"\nDone: {total} total changes across {len(LAYER_PATHS)} layer(s)")
else:
    print("\nNo changes needed.")

simulation_app.close()
