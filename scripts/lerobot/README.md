

## Lerobot数据格式补充
``` bash
pip install lerobot 

python scripts/lerobot/convert_to_lerobot.py \
    --src-dir /workspace/isaaclab/datasets/simdata/V1/SIM-PIPER-GRAB-0622-N100-K-V1 \
    --output-dir /workspace/isaaclab/datasets/lerobot/piper_grab_V1_D3
```

## 训练


## 专家数据转lerobot
``` bash
./isaaclab.sh -p scripts/lerobot/convert_to_lerobot.py \
  --task Isaac-Piper-Grab-IK-Rel-Visuomotor-v1 \
  --input ./datasets/hdf5/[0629]annotated_piper_dataset_K.hdf5 \
  --output ./datasets/lerobot/E-SIM-PIPER-GRAB-0629-N20-K \
  --headless --enable_cameras --device cuda:0
```
