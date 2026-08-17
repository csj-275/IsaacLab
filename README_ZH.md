
# 安装
``` bash
git clone https://github.com/csj-275/IsaacLab.git
cd Isaaclab
git checkout company
git submodule update --init --recursive
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run 
```

# 容器
``` bash 
# 0. 获取权限
newgrp docker
# 1. 启动容器
./docker/container.py start base --files docker-compose-csj.yaml
# 2. 进入容器
./docker/container.py enter base
# 3. 重启容器
docker restart isaac-lab-base`
# cuda设置
export CUDA_VISIBLE_DEVICES=3
```

# 容器中安装Lerobo（4090）
``` bash
pip install lerobot
rm -rf /workspace/isaaclab/_isaac_sim/kit/python/lib/python3.11/site-packages/pip /workspace/isaaclab/_isaac_sim/kit/python/lib/python3.11/site-packages/pip-*.dist-info
curl -sS https://bootstrap.pypa.io/get-pip.py | /workspace/isaaclab/_isaac_sim/python.sh
/workspace/isaaclab/_isaac_sim/python.sh -m pip install numpy==1.26.4
```

# 容器中安装curobo
``` bash
sh /workspace/isaaclab/cuda_12.8.0_570.86.10_linux.run --silent --toolkit --override --toolkitpath=/usr/local/cuda-12.8
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.0+PTX"
export SETUPTOOLS_SCM_PRETEND_VERSION=0.7.7
./_isaac_sim/python.sh -m pip install -e src/nvidia-curobo --no-build-isolation
# 验证
./_isaac_sim/python.sh -c "import curobo; print(curobo.__version__)"
```


-----------------------------------------
# Curobo安装(4090)
## 1. 安装 CUDA 12.8 Toolkit
``` bash
sh /workspace/isaaclab/cuda_12.8.0_570.86.10_linux.run --silent --toolkit --override --toolkitpath=/usr/local/cuda-12.8
```
## 2. 修复 libcusparseLt.so.0 缺失
``` bash
# IsaacSim自带的torch缺少cuSPARSELt稀疏矩阵库，需要软链
TORCH_LIB="/isaac-sim/kit/python/lib/python3.11/site-packages/torch/lib"
CUSPARSELT_SO="/isaac-sim/kit/python/lib/python3.11/site-packages/nvidia/cusparselt/lib/libcusparseLt.so.0"
ln -sf "$CUSPARSELT_SO" "$TORCH_LIB/libcusparseLt.so.0"
```
## 3. 安装 curobo
``` bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.0+PTX"
export SETUPTOOLS_SCM_PRETEND_VERSION=0.7.7
./_isaac_sim/python.sh -m pip install -e src/nvidia-curobo --no-build-isolation
./_isaac_sim/python.sh -m pip install rerun-sdk==0.23
```
## 4. 验证
``` bash
# 验证 curobo torch正常
./_isaac_sim/python.sh -c "import curobo; import torch; 
print(torch.__version__);
print(curobo.__version__);"
```

> **注意**：`libcusparseLt.so.0` 软链在容器重启后会丢失，需要重新执行步骤 2。
----------------------------------------------


# Mimic
``` bash
# 1. 采集数据 - 基础环境
./isaaclab.sh -p scripts/tools/record_demos.py \
--task Isaac-Piper-Grab-IK-Rel-Mimic-v1  \
--teleop_device keyboard \
--dataset_file ./datasets/hdf5/piper_dataset.hdf5 \
--num_demos 3
# 2. 标注子信号 - 基础环境
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
--task Isaac-Piper-Grab-IK-Rel-Mimic-v1 \
--input_file ./datasets/hdf5/piper_dataset.hdf5 \
--output_file ./datasets/hdf5/annotated_piper_dataset.hdf5 \
--device cuda:0 --auto --headless
# 3. 生成Lerobot数据
./isaaclab.sh -p \
scripts/imitation_learning/isaaclab_mimic/generate_dataset_lerobot.py \
--task Isaac-Piper-Grab-IK-Rel-Visuomotor-Mimic-v1 \
--input_file ./datasets/hdf5/annotated_piper_dataset.hdf5 \
--output_file ./datasets/lerobot/lerobot_generated \
--generation_num_trials 10 \
--headless --enable_cameras --device cuda:0 \
--fps 30
# 4. 转完整Lerobot
./isaaclab.sh -p scripts/lerobot/convert_to_lerobot.py --src-dir ./datasets/lerobot/SIM-PIPER-GRAB-0808-N100-L1-V1 --output ./datasets/lerobot/D-SIM-PIPER-GRAB-0808-N100-L1-V1 --use-state-as-action
# 5. 复制Lerobot
cp -r datasets/lerobot/XXXX /home/dgrlab04/csj_ws/lerobot/datasets/

# 6. 数据集合并
python scripts/lerobot/merge_lerobot_datasets.py --input1 datasets/lerobot/XXXX/ --input2 datasets/lerobot/XXXX/ --output /home/dgrlab04/csj_ws/lerobot/datasets/XXXX

```

# MSTSC连接无画面
``` bash
# 1. 在宿主机设置本地 DISPLAY，然后重启容器
export DISPLAY=:10
./docker/container.py stop base
./docker/container.py start base --files docker-compose-csj.yaml
# 2. 进容器确认
./docker/container.py enter base
echo $DISPLAY
# 应该输出 :10
```

# ACT策略测试
``` bash
./isaaclab.sh -p scripts/lerobot/eval_policy.py --checkpoint-dir ./logs/policy/D-SIM-PIPER-GRAB-0808-N100-L1-V1-ACT/checkpoints/last/pretrained_model/ --num-episodes 50 --max-steps 1200 --enable_cameras --device cuda:0 --headless --post-success-delay 60  --video logs/eval_videos
```
