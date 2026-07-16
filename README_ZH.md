

# 容器
**1. 启动容器**
``` bash 
./docker/container.py start base --files docker-compose.xrobotoolkit.patch.yaml
```

**2. 进入容器**
``` bash
./docker/container.py enter base
```

**3. 重启容器**
``` bash
docker restart isaac-lab-base`
```

## 容器中lerobo安装
``` bash
pip install lerobot
rm -rf /workspace/isaaclab/_isaac_sim/kit/python/lib/python3.11/site-packages/pip /workspace/isaaclab/_isaac_sim/kit/python/lib/python3.11/site-packages/pip-*.dist-info
curl -sS https://bootstrap.pypa.io/get-pip.py | /workspace/isaaclab/_isaac_sim/python.sh
/workspace/isaaclab/_isaac_sim/python.sh -m pip install numpy==1.26.4
```

## 容器中cuRobo安装
``` bash
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run 
sh cuda_12.8.0_570.86.10_linux.run --silent --toolkit --override --toolkitpath=/usr/local/cuda-12.8

export CUDA_HOME=/usr/local/cuda-12.8 && \
export PATH="$CUDA_HOME/bin:$PATH" && \
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH" && \
export TORCH_CUDA_ARCH_LIST="8.0+PTX"

git clone https://github.com/NVlabs/curobo.git /src/nvidia-curobo
cd /src/nvidia-curobo
git checkout ebb71702f3f70e767f40fd8e050674af0288abe8
cd ..
cd ..
pip install -e src/nvidia-curobo --no-build-isolation

```