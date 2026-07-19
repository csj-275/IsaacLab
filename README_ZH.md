
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

# 容器中安装Lerobo（如果需要）
``` bash
pip install lerobot
pip install numpy==1.26.4
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

> 适用于 RTX 4090 (CUDA 12.8)，解决 `libcusparseLt.so.0: cannot open shared object file` 错误。

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
--input_file ./datasets/simdata/V1/piper_dataset.hdf5 \
--output_file ./datasets/simdata/V1/annotated_piper_dataset.hdf5 \
--device cuda:0 --auto --headless
# 3. 生成Lerobot数据
./isaaclab.sh -p \
scripts/imitation_learning/isaaclab_mimic/generate_dataset_lerobot.py \
--task Isaac-Piper-Grab-IK-Rel-Visuomotor-Mimic-v1 \
--input_file ./datasets/hdf5/annotated_piper_dataset.hdf5 \
--output_file ./datasets/simdata/V1/lerobot_generated \
--generation_num_trials 10 \
--headless --enable_cameras --device cuda:0 \
--fps 30
```

-----------------------------------------------------

# Skillgen
``` bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
--device cpu \
--num_envs 1 \
--generation_num_trials 10 \
--input_file ./datasets/hdf5/annotated_dataset_skillgen.hdf5 \
--output_file ./datasets/hdf5/generated_dataset_skillgen.hdf5 \
--task Isaac-Piper-Grab-IK-Rel-Skillgen-v1 \
--use_skillgen

```

-----------------------------------------------------
# ACT策略测试
``` bash
./isaaclab.sh -p scripts/lerobot/eval_policy.py \
    --checkpoint-dir ./logs/policy/D-SIM-PIPER-GRAB-0702-N100-K-V1-ACT/checkpoints/last/pretrained_model/ \
    --num-episodes 1 \
    --max-steps 800 \
    --enable_cameras \
    --device cuda:0 --headless \
    --video ./log/eval_videos
```

-----------------------------------------

**常见报错**：ModuleNotFoundError: No module named 'pip._vendor.packaging._structures'
There was an error running python
``` bash
rm -rf /workspace/isaaclab/_isaac_sim/kit/python/lib/python3.11/site-packages/pip /workspace/isaaclab/_isaac_sim/kit/python/lib/python3.11/site-packages/pip-*.dist-info
curl -sS https://bootstrap.pypa.io/get-pip.py | /workspace/isaaclab/_isaac_sim/python.sh
/workspace/isaaclab/_isaac_sim/python.sh -m pip install numpy==1.26.4
```