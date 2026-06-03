"""Verify bottle.usd and box.usd are clean of articulation APIs."""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
app = AppLauncher(parser.parse_args([])).app

from pxr import Usd, UsdPhysics

BAD_APIS = {"IsaacRobotAPI", "IsaacLinkAPI", "PhysicsArticulationRootAPI", "PhysxArticulationAPI"}

for name in ["bottle", "box"]:
    stage = Usd.Stage.Open("usd/%s/%s.usd" % (name, name))
    issues = []
    for prim in stage.TraverseAll():
        apis = list(prim.GetAppliedSchemas())
        for api in apis:
            if api in BAD_APIS:
                issues.append("%s: %s" % (prim.GetPath(), api))
        if prim.IsA(UsdPhysics.FixedJoint) and prim.IsActive():
            issues.append("%s: active FixedJoint" % prim.GetPath())

    if issues:
        print("%s: ISSUES: %s" % (name, issues))
    else:
        print("%s: CLEAN" % name)

# Also write to file for reliable capture
import os
with open(os.path.join(os.path.dirname(__file__), "verify_usd_fix_result.txt"), "w") as f:
    for name in ["bottle", "box"]:
        stage = Usd.Stage.Open("usd/%s/%s.usd" % (name, name))
        issues = []
        for prim in stage.TraverseAll():
            apis = list(prim.GetAppliedSchemas())
            for api in apis:
                if api in BAD_APIS:
                    issues.append("%s: %s" % (prim.GetPath(), api))
            if prim.IsA(UsdPhysics.FixedJoint) and prim.IsActive():
                issues.append("%s: active FixedJoint" % prim.GetPath())
        f.write("%s: %s\n" % (name, ("ISSUES:" + str(issues)) if issues else "CLEAN"))

app.close()
