"""Convert HDF5 demo dataset to MP4 videos for VLA training.

Usage:
    # Convert all demos, all cameras
    python datasets/hdf5_to_mp4.py --input ./datasets/simdata/V1/visuo_dataset.hdf5 --output_dir ./datasets/videos

    # Specific demos and cameras
    python datasets/hdf5_to_mp4.py --input ./datasets/simdata/V1/visuo_dataset.hdf5 --demos demo_0,demo_1 --cameras table_cam
"""

import argparse
import h5py
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Convert HDF5 dataset to MP4 videos")
    parser.add_argument("--input", type=str, required=True, help="Input HDF5 file path")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: next to input)")
    parser.add_argument("--demos", type=str, default="", help="Comma-separated demo keys, e.g. demo_0,demo_1. Empty = all")
    parser.add_argument("--cameras", type=str, default="table_cam,wrist_cam", help="Comma-separated camera names")
    parser.add_argument("--fps", type=int, default=15, help="Output video FPS")
    args = parser.parse_args()

    input_path = Path(args.input)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_path.parent.parent / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)

    demo_keys = [d.strip() for d in args.demos.split(",") if d.strip()] if args.demos else []
    cameras = [c.strip() for c in args.cameras.split(",")]

    with h5py.File(input_path, "r") as f:
        available = [k for k in f["data"].keys() if k.startswith("demo_")]
        available.sort(key=lambda x: int(x.split("_")[1]))
        demo_keys = demo_keys or available

        for demo_name in demo_keys:
            demo_path = f"data/{demo_name}" if not demo_name.startswith("data/") else demo_name
            demo_short = demo_path.split("/")[-1]

            if demo_path not in f:
                print(f"  [{demo_short}]: not found, skip")
                continue

            demo = f[demo_path]
            for cam in cameras:
                obs_path = f"obs/{cam}"
                if obs_path not in demo:
                    obs_path_alt = f"obs/{cam}"
                    found = False
                    # try to find camera in obs group
                    for k in demo.get("obs", {}).keys():
                        if cam in k:
                            obs_path = f"obs/{k}"
                            found = True
                            break
                    if not found:
                        print(f"  [{demo_short}] {cam}: not found, skip")
                        continue

                frames = demo[obs_path][:]
                T, H, W, C = frames.shape
                output = output_dir / f"{demo_short}_{cam}.mp4"

                cmd = [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "rawvideo", "-vcodec", "rawvideo",
                    "-s", f"{W}x{H}", "-pix_fmt", "rgb24", "-r", str(args.fps),
                    "-i", "-",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                    "-pix_fmt", "yuv420p",
                    str(output),
                ]
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
                proc.communicate(frames.tobytes())

                size_mb = output.stat().st_size / 1024 / 1024
                print(f"  [{demo_short}] {cam} -> {output} ({size_mb:.1f} MB, {T} frames)")

    print("Done.")


if __name__ == "__main__":
    main()
