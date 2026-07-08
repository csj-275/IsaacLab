

## Lerobot数据格式补充
``` bash
pip install lerobot 

python scripts/reconova/lerobot/convert_to_lerobot.py \
    --src-dir ./datasets \
    --output-dir ./datasets/lerobot/piper_grab_v1
```

## 训练
python 

## 专家数据转lerobot
``` bash
./isaaclab.sh -p scripts/reconova/lerobot/convert_to_lerobot.py \
  --task Isaac-Piper-Grab-IK-Rel-Visuomotor-v1 \
  --input ./datasets/piper_annotated_dataset.hdf5 \
  --output ./datasets/lerobot/piper_grab_v1 \
  --headless --enable_cameras --device cuda:0
```
