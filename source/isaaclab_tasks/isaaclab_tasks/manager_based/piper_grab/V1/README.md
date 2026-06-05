# V1 
**0.容器启动**
``` bash
  ./docker/container.py start base --files docker-compose.xrobotoolkit.patch.yaml # 启动容器
  ./docker/container.py enter base # 进入容器
  docker restart $(docker ps -q) # 重启容器，Isaacsim卡住时使用
```

-----------------------------------------------------
- 主要使用的版本为IK, Visuo
- Cosmos一般不使用

**A1. IK录制演示**
``` bash
./isaaclab.sh -p scripts/tools/record_demos.py --task Isaac-Piper-Grab-IK-Rel-Mimic-v1 --device cpu --teleop_device keyboard --dataset_file ./datasets/simdata/V1/piper_dataset.hdf5 --num_demos 1
```

**A2. IK自动标注**
``` bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py --auto --device cpu --task Isaac-Piper-Grab-IK-Rel-Mimic-v1 --input_file ./datasets/piper_dataset.hdf5 --output_file ./datasets/annotated_piper_dataset.hdf5
```

-----------------------------------------

**B1. 视觉录制演示**
```bash
./isaaclab.sh -p scripts/tools/record_demos.py --task Isaac-Piper-Grab-IK-Rel-Visuomotor-Mimic-v1 --teleop_device keyboard --dataset_file ./datasets/simdata/V1/visuo_dataset.hdf5 --num_demos 1 --enable_cameras
```

**B2. 视觉自动标注**
```bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py  --task Isaac-Piper-Grab-IK-Rel-Visuomotor-Mimic-v1 --input_file ./datasets/simdata/V1/visuo_dataset.hdf5 --output_file ./datasets/simdata/V1/annotated_visuo_dataset.hdf5 --enable_cameras --device cpu --headless --auto
```

**B3. 视觉生成数据**
``` bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py --generation_num_trials 10 --input_file ./datasets/simdata/V1/annotated_visuo_dataset.hdf5 --output_file ./datasets/simdata/V1/generated_visuo_dataset.hdf5 --headless --enable_cameras
```

-----------------------------------------------
**C1. Cosmos录制演示**
```bash
    ./isaaclab.sh -p scripts/tools/record_demos.py --task Isaac-Piper-Grab-IK-Rel-Visuomotor-Cosmos-Mimic-v1 --device cpu --teleop_device keyboard --dataset_file ./datasets/cosmos_dataset.hdf5 --num_demos 10 --enable_cameras
```