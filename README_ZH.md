
# 安装
``` bash
git clone https://github.com/csj-275/IsaacLab.git
cd Isaaclab
git checkout company
git submodule update --init --recursive
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run 
```

# 容器
权限获取：`newgrp docker`
``` bash 
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

---














## 容器中lerobo安装
``` bash
pip install lerobot
rm -rf /workspace/isaaclab/_isaac_sim/kit/python/lib/python3.11/site-packages/pip /workspace/isaaclab/_isaac_sim/kit/python/lib/python3.11/site-packages/pip-*.dist-info
curl -sS https://bootstrap.pypa.io/get-pip.py | /workspace/isaaclab/_isaac_sim/python.sh
/workspace/isaaclab/_isaac_sim/python.sh -m pip install numpy==1.26.4
```

## 容器中cuRobo安装
``` bash
sh cuda_12.8.0_570.86.10_linux.run --silent --toolkit --override --toolkitpath=/usr/local/cuda-12.8 && \
export CUDA_HOME=/usr/local/cuda-12.8 && \
export PATH="$CUDA_HOME/bin:$PATH" && \
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH" && \
export TORCH_CUDA_ARCH_LIST="8.0+PTX"
pip install -e src/nvidia-curobo --no-build-isolation

```

## 容器中cuRobo安装（修正版 2026-07-16）

上面旧版的三个问题：
1. 裸 `pip` 缺少 Isaac Sim 的环境变量（PYTHONPATH），torch 找不到预置的 CUDA 库，报 `libcusparseLt.so.0`——必须用 `python.sh -m pip`；
2. `export LD_LIBRARY_PATH=$CUDA_HOME/lib64:...` 会让 torch 跳过预置库的预加载，报 `libcudnn.so.9`——这行不能加；
3. curobo 以子模块挂载进容器，容器内 `.git` 指针无法解析，setuptools-scm 取不到版本号——需用环境变量指定。

``` bash
# 1. 安装 CUDA toolkit(安装包已由 docker-compose-csj.yaml 挂载进容器;容器重建后需重跑本节全部步骤)
sh /workspace/isaaclab/cuda_12.8.0_570.86.10_linux.run --silent --toolkit --override --toolkitpath=/usr/local/cuda-12.8

# 2. 编译环境(注意:不要设置 LD_LIBRARY_PATH)
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.0+PTX"
export SETUPTOOLS_SCM_PRETEND_VERSION=0.7.7

# 3. 用 Isaac Sim 的 python 安装(不要用裸 pip)
cd /workspace/isaaclab
./_isaac_sim/python.sh -m pip install -e src/nvidia-curobo --no-build-isolation

# 4. 验证
./_isaac_sim/python.sh -c "import curobo; print(curobo.__version__)"
```
