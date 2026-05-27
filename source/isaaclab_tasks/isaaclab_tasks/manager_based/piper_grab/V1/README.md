# V1 

-----------------------------------------------------

**A1. IK录制演示**
``` bash
./isaaclab.sh -p scripts/tools/record_demos.py --task Isaac-Piper-Grab-IK-Rel-Mimic-v1 --device cpu --teleop_device keyboard --dataset_file ./datasets/piper_dataset.hdf5 --num_demos 1
```

**A2. IK自动标注**
``` bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py --auto --device cpu --task Isaac-Piper-Grab-IK-Rel-Mimic-v1 --input_file ./datasets/piper_dataset.hdf5 --output_file ./datasets/annotated_piper_dataset.hdf5
```

-----------------------------------------

**B1. 视觉录制演示**
```bash
./isaaclab.sh -p scripts/tools/record_demos.py --task Isaac-Piper-Grab-IK-Rel-Visuomotor-Mimic-v1 --device cpu --teleop_device keyboard --dataset_file /data/simdata/V1/visuo_dataset.hdf5 --num_demos 20 --enable_cameras
```

**B2. 视觉自动标注**
```bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py --auto --device cpu --task Isaac-Piper-Grab-IK-Rel-Visuomotor-Mimic-v1 --input_file /data/simdata/V1/visuo_dataset.hdf5 --output_file /data/simdata/V1/annotated_visuo_dataset.hdf5 --enable_cameras --headless
```

**B3. 视觉生成数据**
``` bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py --generation_num_trials 10 --input_file /data/simdata/V1/annotated_visuo_dataset.hdf5 --output_file /data/simdata/V1/generated_visuo_dataset.hdf5 --headless --enable_cameras
```

-----------------------------------------------
**C1. Cosmos录制演示**
```bash
    ./isaaclab.sh -p scripts/tools/record_demos.py --task Isaac-Piper-Grab-IK-Rel-Visuomotor-Cosmos-Mimic-v1 --device cpu --teleop_device keyboard --dataset_file ./datasets/cosmos_dataset.hdf5 --num_demos 10 --enable_cameras
```