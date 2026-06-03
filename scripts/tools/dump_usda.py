"""Export a USD layer to USDA text format for inspection.
Usage: isaaclab -p scripts/tools/dump_usda.py --headless <input.usd> <output.usda>
"""
import argparse, os, sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Export USD layer to USDA text format")
parser.add_argument("input", type=str, help="Input USD file (USDC or USDA)")
parser.add_argument("output", type=str, help="Output USDA file")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

from pxr import Sdf

inpath = os.path.abspath(args_cli.input)
outpath = os.path.abspath(args_cli.output)

layer = Sdf.Layer.FindOrOpen(inpath)
if not layer:
    print(f"ERROR: Cannot open {inpath}")
    simulation_app.close()
    sys.exit(1)

layer.Export(outpath)
print(f"OK: {inpath} -> {outpath}")
print(f"    Lines: {sum(1 for _ in open(outpath))}")
simulation_app.close()
