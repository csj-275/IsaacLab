"""Debug: print prim-to-layer mapping for bottle.usd."""
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args([])
app = AppLauncher(args_cli).app

from pxr import Usd, Sdf, UsdGeom, UsdPhysics

USD_PATH = "usd/bottle/bottle.usd"
stage = Usd.Stage.Open(USD_PATH)

sys.stdout.write("=== Prim -> Layer mapping ===\n")
sys.stdout.flush()

for prim in stage.TraverseAll():
    path = str(prim.GetPath())
    if "bottle" in path.lower():
        # Get the layer stack for this prim
        prim_stack = prim.GetPrimStack()
        layer_id = "???"
        if prim_stack:
            layer_id = str(prim_stack[0].layer.identifier) if hasattr(prim_stack[0], 'layer') else "?"
        # Get the owning layer
        owning_layer = "???"
        try:
            scene_desc = prim.GetSceneDescription()
        except:
            scene_desc = None
        apis = prim.GetAppliedSchemas()
        sys.stdout.write(f"{path} [{prim.GetTypeName()}] apis={apis}\n")
        sys.stdout.write(f"  specifier={prim.GetSpecifier()}\n")
        if prim.GetSpecifier() == Sdf.SpecifierDef:
            sys.stdout.write(f"  -> DEFINED here\n")
        elif prim.GetSpecifier() == Sdf.SpecifierOver:
            sys.stdout.write(f"  -> OVERRIDE\n")
        sys.stdout.flush()

sys.stdout.write("\n=== Checking layers directly ===\n")
sys.stdout.flush()

# Open the referenced layer directly
import os
REF_PATH = os.path.join(os.path.dirname(os.path.abspath(USD_PATH)), "urdf/bottle/bottle.usd")
sys.stdout.write(f"Opening referenced layer: {REF_PATH}\n")
sys.stdout.flush()
ref_stage = Usd.Stage.Open(REF_PATH)
for prim in ref_stage.TraverseAll():
    path = str(prim.GetPath())
    apis = prim.GetAppliedSchemas()
    if "bottle" in path.lower() or "joint" in path.lower() or "link" in path.lower():
        sys.stdout.write(f"REF_LAYER {path} [{prim.GetTypeName()}] apis={apis}\n")
        sys.stdout.flush()

app.close()
