# LeRobot ACT 训练流程

将 IsaacLab `simdata` 数据转换为 [LeRobot](https://github.com/huggingface/lerobot) v3.0 格式，并使用 ACT（Action Chunking Transformer）算法训练策略。

## 目录结构

```
isaaclab/
├── lerobot_train/                          # 本文档目录
│   └── README.md                           # 本文件
├── scripts/
│   ├── convert_to_lerobot.py               # 数据格式转换脚本
│   └── train_act_piper_grab.py             # ACT 训练脚本
├── datasets/
│   ├── simdata/V1/SIM-*/                   # IsaacLab 原始数据（输入）
│   └── lerobot/
│       ├── piper_grab_v1/                  # 转换后的 LeRobot 数据集
│       │   ├── data/chunk-000/*.parquet    # 动作/状态数据
│       │   ├── videos/
│       │   │   ├── observation.images.front/chunk-000/*.mp4
│       │   │   └── observation.images.wrist/chunk-000/*.mp4
│       │   └── meta/
│       │       ├── info.json               # 数据集元信息
│       │       ├── stats.json              # 归一化统计量
│       │       ├── tasks.parquet           # 语言任务表
│       │       └── episodes/               # episode 索引
│       └── piper_grab_v1_act_checkpoints/  # ACT 训练 checkpoint
└── docker/
    └── docker-compose.xrobotoolkit.patch.yaml  # 挂载 lerobot + datasets
```

## 文件说明

### 1. `scripts/convert_to_lerobot.py` — 数据格式转换

将 IsaacLab simdata 目录转换为 LeRobot v3.0 格式。

**IsaacLab 原始数据要求：**
```
simdata/V1/<NAME>/
├── data/chunk-000/
│   └── file-*.parquet        # 含 action, observation.state, timestamp, frame_index,
│                             #   episode_index, index, task_index 列
├── videos/
│   ├── observation.images.<CAM>/chunk-000/file-*.mp4   # 每 episode 一个 mp4
│   └── observation.depths.<CAM>/...                    # 深度（可选，不转换）
└── meta/episodes/...         # 可选
```

**用法：**
```bash
# 基本用法（在主机上）
conda activate lerobot_py312
python scripts/convert_to_lerobot.py \
    --src-dir datasets/simdata/V1/SIM-PIPER-GRAB-0618-N100-IK-K-V1 \
    --output-dir datasets/lerobot/piper_grab_v1

# 完整参数
python scripts/convert_to_lerobot.py \
    --src-dir /path/to/isaaclab_simdata \
    --output-dir /path/to/lerobot_dataset \
    --symlink    # 使用符号链接而非拷贝视频（节省空间）
```

**关键常量（如数据维度变化需修改）：**
- `build_features()` — action/state 维度、相机名称和分辨率
- `"tasks": [...]` — 语言任务描述
- `fps=30` — 采集帧率

### 2. `scripts/train_act_piper_grab.py` — ACT 策略训练

使用 LeRobot ACT 算法训练抓取策略。

**功能：**
- 自动检测容器内/外路径
- 双路相机输入（front + wrist）+ 机器人状态 (63-dim)
- 输出 8 维 IK 空间动作
- VAE + ResNet18 视觉 backbone + Transformer
- 定期保存 checkpoint

**用法：**
```bash
# 容器内（推荐，有 GPU 加速）
cd /workspace/isaaclab
CUDA_VISIBLE_DEVICES=0 python scripts/train_act_piper_grab.py

# 主机上（需 conda lerobot_py312 环境）
conda activate lerobot_py312
cd /home/chenshengjia/company/isaaclab
python scripts/train_act_piper_grab.py
```

**可调参数（脚本顶部）：**
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `BATCH_SIZE` | 8 | 批次大小 |
| `NUM_EPOCHS` | 500 | 训练轮数 |
| `CHUNK_SIZE` | 50 | 预测未来 action 帧数 |
| `IMG_RESIZE` | (224, 224) | 图像 resize 尺寸 |
| `LR` | 1e-5 | 学习率 |

**输出：**
- `datasets/lerobot/piper_grab_v1_act_checkpoints/checkpoint_step_XXXXXX/` — 定期保存
- `datasets/lerobot/piper_grab_v1_act_checkpoints/final/` — 最终模型

## 标准流程

### Step 1: 准备 LeRobot 环境（主机，仅需一次）

```bash
# 创建 Python 3.12 环境
conda create -n lerobot_py312 python=3.12 -y
conda activate lerobot_py312

# 安装 LeRobot（从本地 git 仓库）
cd /home/chenshengjia/company/lerobot
pip install -e ".[dataset]"
```

### Step 2: 转换数据

有新数据时执行此步。以 `SIM-PIPER-GRAB-0618-N100-IK-K-V1` 为例：

```bash
conda activate lerobot_py312
cd /home/chenshengjia/company/isaaclab

python scripts/convert_to_lerobot.py \
    --src-dir datasets/simdata/V1/SIM-PIPER-GRAB-0618-N100-IK-K-V1 \
    --output-dir datasets/lerobot/piper_grab_v1
```

> **注意：** 如果数据维度或相机配置变了，需要修改 `convert_to_lerobot.py` 中的 `build_features()` 函数。

### Step 3: 在 Docker 容器内训练

```bash
# 启动容器
cd /home/chenshengjia/company/isaaclab
./docker/container.py start base --files docker-compose.xrobotoolkit.patch.yaml

# 进入容器
./docker/container.py enter base

# 容器内首次：安装 LeRobot
pip install -e "/workspace/lerobot[dataset]"

# 运行训练
CUDA_VISIBLE_DEVICES=0 python /workspace/isaaclab/scripts/train_act_piper_grab.py
```

### Step 4: 查看结果

Checkpoint 保存在 `datasets/lerobot/piper_grab_v1_act_checkpoints/`，可同时在容器内和主机访问。

## 容器内路径映射

| 主机路径 | 容器内路径 |
|---|---|
| `isaaclab/datasets/` | `/workspace/isaaclab/datasets/` |
| `isaaclab/scripts/` | `/workspace/isaaclab/scripts/` |
| `~/company/lerobot/` | `/workspace/lerobot/` |

## 适配新数据

当生成新的 simdata 时，需要修改以下内容：

**`convert_to_lerobot.py` — `build_features()`:**
```python
return {
    "action": {"dtype": "float32", "shape": (8,), ...},      # 改 action 维度
    "observation.state": {"dtype": "float32", "shape": (63,), ...},  # 改 state 维度
    "observation.images.front": {..., "shape": (H, W, 3), ...},   # 改分辨率
    "observation.images.wrist": {..., "shape": (H, W, 3), ...},
}
```

**`convert_to_lerobot.py` — 任务描述:**
```python
# 行 127 附近
"tasks": ["your task description here"],
```

**`train_act_piper_grab.py` — `DATASET_ROOT`:**
- 脚本自动检测容器内外路径，如新增路径加到 `_find_dataset_root()` 的 `candidates` 列表。

**`train_act_piper_grab.py` — 训练超参数:**
- 修改脚本顶部的 `BATCH_SIZE`、`NUM_EPOCHS`、`CHUNK_SIZE` 等。

## 已知问题

1. **LeRobot 本地加载需联网？** — 代码会尝试访问 HuggingFace Hub 验证版本号。已在训练脚本中 monkey-patch `get_safe_version` 跳过此检查。若手动加载数据集，需执行相同 patch。

2. **`observation.state` 的 delta_timestamps** — ACT 内部自己管理观测时序（`n_obs_steps=1`），不要在 `delta_timestamps` 中包含 `observation.state`。

3. **视频解码依赖** — 需要安装 `torchcodec` 或 `av`（已包含在 `lerobot[dataset]` 中）。

4. **Windows 远程连接进入容器报 `KeyError: 'DISPLAY'`** — 已在 `docker/utils/x11_utils.py` 修复：`DISPLAY` 不存在时跳过 X11 凭据刷新，不再崩溃。如遇到旧版本，`export DISPLAY=:0` 临时绕过。
