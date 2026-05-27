#!/usr/bin/env python3
"""Add PhysicsRigidBodyAPI to mug.usd so it can be used as a RigidObjectCfg.

Run inside the Isaac Sim container:
    isaaclab --python scripts/tools/fix_mug_usd.py
"""

import os
from pxr import Sdf, Usd, UsdPhysics, UsdGeom

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "../.."))
MUG_PATH = os.path.join(REPO_ROOT, "usd/mug.usd")


def main():
    print(f"Opening: {MUG_PATH}")
    # Open the root layer directly for editing
    layer = Sdf.Layer.FindOrOpen(MUG_PATH)
    if not layer:
        raise RuntimeError(f"Failed to open: {MUG_PATH}")

    # Open stage with this layer as root layer
    stage = Usd.Stage.Open(layer.identifier)
    # Redirect edits to the actual file layer (not the anonymous session layer)
    stage.SetEditTarget(layer)

    # Inspect prim structure
    print("Prims in stage:")
    for prim in stage.TraverseAll():
        print(f"  {prim.GetPath()} [{prim.GetTypeName()}]")
        for api in prim.GetAppliedSchemas():
            print(f"    api: {api}")

    # Find mesh prims and add physics APIs
    needs_fix = False
    for prim in stage.TraverseAll():
        if prim.IsA(UsdGeom.Mesh):
            print(f"\nProcessing mesh: {prim.GetPath()}")

            if not UsdPhysics.RigidBodyAPI.Apply(prim):
                print("  -> Already has PhysicsRigidBodyAPI")
            else:
                needs_fix = True
                print("  -> Added PhysicsRigidBodyAPI")
                mass_api = UsdPhysics.MassAPI.Apply(prim)
                mass_api.GetMassAttr().Set(0.1)
                print("  -> Added MassAPI (0.1 kg)")

            if not UsdPhysics.CollisionAPI.Apply(prim):
                print("  -> Already has CollisionAPI")
            else:
                needs_fix = True
                print("  -> Added CollisionAPI")

    if needs_fix:
        layer.Save()
        print(f"\nSaved to: {MUG_PATH}")
        # Verify
        stage2 = Usd.Stage.Open(MUG_PATH)
        print("Verification:")
        for prim in stage2.TraverseAll():
            if prim.IsA(UsdGeom.Mesh):
                has_rb = prim.HasAPI(UsdPhysics.RigidBodyAPI)
                has_col = prim.HasAPI(UsdPhysics.CollisionAPI)
                print(f"  {prim.GetPath()}: rigid_body={has_rb}, collision={has_col}")
    else:
        print("\nNo changes needed.")


if __name__ == "__main__":
    main()
