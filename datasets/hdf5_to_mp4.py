import h5py
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent  # datasets/ 目录

# ============================================================
# 配置参数 - 需修改
# ============================================================
HDF5_FILE = SCRIPT_DIR / "simdata/V1/visuo_dataset.hdf5"
# OUTPUT_DIR = "/data/chenshengjia/simdata/V1/videos"
OUTPUT_DIR = "./datasets/videos"
DEMO_KEYS = []  # 留空 [] 表示全部 demo，或指定如 ["data/demo_0", "data/demo_1"]
CAMERAS = ["table_cam", "wrist_cam"]  # 可选: table_cam, wrist_cam
FPS = 15
# ============================================================

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

with h5py.File(HDF5_FILE, "r") as f:
    available = [k for k in f["data"].keys() if k.startswith("demo_")]
    available.sort(key=lambda x: int(x.split("_")[1]))

    demo_keys = DEMO_KEYS if DEMO_KEYS else [f"data/{d}" for d in available]

    for demo_key in demo_keys:
        demo_name = demo_key.split("/")[-1]
        demo = f[demo_key]

        for cam in CAMERAS:
            if cam not in demo["obs"]:
                print(f"  [{demo_name}] {cam}: not found, skip")
                continue

            frames = demo[f"obs/{cam}"][:]
            T, H, W, C = frames.shape
            output = f"{OUTPUT_DIR}/{demo_name}_{cam}.mp4"

            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "rawvideo", "-vcodec", "rawvideo",
                "-s", f"{W}x{H}", "-pix_fmt", "rgb24", "-r", str(FPS),
                "-i", "-",
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-pix_fmt", "yuv420p",
                output,
            ]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            proc.communicate(frames.tobytes())

            size_mb = Path(output).stat().st_size / 1024 / 1024
            print(f"  [{demo_name}] {cam} -> {output} ({size_mb:.1f} MB, {T} frames)")

print("Done.")
