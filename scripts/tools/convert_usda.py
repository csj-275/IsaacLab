"""Convert a USDA text file to USDC binary format.
Usage: isaaclab -p scripts/tools/convert_usda.py --headless <input.usda> <output.usdc>
"""
import argparse, os, sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Convert USDA text to USDC binary")
parser.add_argument("input", type=str, help="Input USDA text file")
parser.add_argument("output", type=str, help="Output USDC binary file")
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
simulation_app.close()
