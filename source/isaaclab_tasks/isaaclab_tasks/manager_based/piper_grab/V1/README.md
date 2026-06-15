# V1 
## 0.容器启
``` bash
  ./docker/container.py start base --files docker-compose.xrobotoolkit.patch.yaml # 启动容器
  ./docker/container.py enter base # 进入容器
  docker restart isaac-lab-base
 # 重启容器，Isaacsim卡住时使用
```

``` bash
# 设置CUDA使用0,1
    export CUDA_VISIBLE_DEVICES=0,1
    # 取消渲染
    DISPLAY=
    # 查卡显卡使用
    docker exec isaac-lab-base nvidia-smi
```

------------------------------------------------
## IK
### A1. IK录制演示
``` bash
./isaaclab.sh -p scripts/tools/record_demos.py --task Isaac-Piper-Grab-IK-Rel-Mimic-v1 --device cpu --teleop_device keyboard --dataset_file ./datasets/simdata/V1/piper_dataset.hdf5 --num_demos 1
```

### A2. IK自动标注
``` bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py --auto --device cpu --task Isaac-Piper-Grab-IK-Rel-Mimic-v1 --input_file ./datasets/piper_dataset.hdf5 --output_file ./datasets/annotated_piper_dataset.hdf5
```

-----------------------------------------
## Visuo
### B1. 视觉录制演示
```bash
./isaaclab.sh -p scripts/tools/record_demos.py --task Isaac-Piper-Grab-IK-Rel-Visuomotor-Mimic-v1 --teleop_device keyboard --dataset_file ./datasets/simdata/V1/visuo_dataset.hdf5 --num_demos 1 --enable_cameras --device cuda:0
```

### B1-A. 视觉录制演示(XRobo IK-Abs版本)
```bash
bash /workspace/isaaclab/scripts/tools/setup_xrobotoolkit_env.sh

./isaaclab.sh -p scripts/tools/record_demos.py --teleop_device xrobotoolkit \
--xrobotoolkit_mapping_mode world_frame_calibrated \
--xrobotoolkit_calibration_json logs/piper_calibration/xrobotoolkit_world_frame_calibration_20260612_022755.json \
--task Isaac-Piper-Grab-IK-Abs-Visuomotor-Mimic-v1 \
--dataset_file ./datasets/simdata/V1/visuo_dataset.hdf5 --num_demos 20 --enable_cameras --device cuda:0
```

### B2. 视觉自动标注
```bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py  --task Isaac-Piper-Grab-IK-Rel-Visuomotor-Mimic-v1 --input_file ./datasets/simdata/V1/visuo_dataset.hdf5 --output_file ./datasets/simdata/V1/annotated_visuo_dataset.hdf5 --enable_cameras --device cuda:0 --headless --auto
```

### B3. 视觉生成数据
``` bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py --generation_num_trials 10 --input_file ./datasets/simdata/V1/[0611]annotated_visuo_dataset.hdf5 --output_file ./datasets/simdata/V1/[0611]generated_visuo_dataset_N5_V2.hdf5 --enable_cameras --device cuda:0 --task Isaac-Piper-Grab-IK-Rel-Visuomotor-Mimic-v2
```

-----------------------------------------------
## Cosmos
### C1. Cosmos录制演示
```bash
    ./isaaclab.sh -p scripts/tools/record_demos.py --task Isaac-Piper-Grab-IK-Rel-Visuomotor-Cosmos-Mimic-v1 --device cuda:0 --teleop_device keyboard --dataset_file ./datasets/cosmos_dataset.hdf5 --num_demos 10 --enable_cameras
```


## 生成数据(Lerobot版)
``` bash
DISPLAY= CUDA_VISIBLE_DEVICES=1 ./isaaclab.sh -p \
  scripts/imitation_learning/isaaclab_mimic/generate_dataset_lerobot.py \
  --task Isaac-Piper-Grab-IK-Rel-Visuomotor-Mimic-v1 \
  --input_file ./datasets/simdata/V1/[0604]annotated_visuo_dataset.hdf5 \
  --output_file ./datasets/simdata/V1/lerobot_generated_N10 \
  --generation_num_trials 10 \
  --headless --enable_cameras --device cuda:0 \
  --fps 30
```