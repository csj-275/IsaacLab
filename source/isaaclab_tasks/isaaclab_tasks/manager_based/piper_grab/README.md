## 仿真数据生成——任务规划
### 1. 基础环境
1. piper关节空间控制: `piper_env_cfg` - 完成
2. piper任务空间控制: `piper_ik_env_cfg` - 完成
3. grab任务的mdp观测与终止条件: `mdp.observations`, `mdp.terminations` - 完成
4. grab环境: `grab_env_cfg` - 完成 
5. grab环境随机化: `grab_instance_randomize_env_cfg` - 完成  
6. piper关节空间的grab环境： `grab_joint_pos_env_cfg` - 初步完成测试 - offset need to check
7. piper关节空间的grab环境随机化：`grab_joint_pos_instance_randomize_env_cfg` - 完成
8. piper任务空间的grab环境： `grab_ik_rel_env_cfg` - 完成
9. piper任务空间的grab环境随机化：`grab_ik_rel_instance_randomize_env_cfg` - 完成
10. piper视觉驱动的grab环境：`grab_ik_rel_visuomotor_env_cfg` - 完成

### 2. Mimic环境
1. piper mimic环境创建：`piper_grab_ik_rel_mimic_env.py` - 完成
2. piper mimic环境配置：`piper_grab_ik_rel_mimic_env_cfg.py` - 完成
3. piper mimic skillgen环境配置：`piper_grab_ik_rel_skillgen_env_cfg.py` - 完成
4. piper mimic visuomotor环境配置：`piper_grab_ik_rel_visuuomotor_mimic_env_cdf.py` - 完成
- skillgen和mimic都可以生成数据，关键区别在于标注数据时，mimic只需要标注子任务完成，在物体被抓取后立刻标记；skillgen需要标注子任务的开始和结束，即抓取物体时，开始信号在在夹具关闭之前，终止信号是在物体被抓取之后

- skillgen标注需使用`--annotate_subtask_start_signals`
当前情况：仿真遥操作的部分基本完成，现准备接入pico

### 3. Skillgen数据采集、标注与生成
1. piper基于状态的策略环境：`grab_ik_rel_env_cfg_skillgen` - 待做
2. piper基于视觉的策略环境：`grab_ik_rel_visuomotor_env_cfg_skillgen` - 待做
3. 执行以下脚本 (测试并修改Bug)
- piper抓取物体演示数据记录 - 使用`scripts/tools/record_demos.py`
- piper演示数据回放 - 使用`scripts/tools/replay_demos.py`
- piper演示数据标注 - 使用 `scripts/imitation_learning/isaaclab_mimic/annotate_demos.py`
- 数据生成 - 使用 `scripts/imitation_learning/isaaclab_mimic/generate_dataset.py`


**模块关系**
- grab环境包含与机器人构型无关的抓取环境配置，但与待抓取物体个数、类型有关
- 具体机器人继承grab环境，配置robot和ee_frame以及待抓取物体，先关节空间，再任务空间；确定环境和随机环境无继承关系，通常任务空间环境继承自关节空间环境


**1. IK-Rel记录数据**
- `./isaaclab.sh -p scripts/tools/record_demos.py --task Isaac-Piper-Grab-IK-Rel-Mimic-v0 --device cpu --teleop_device keyboard --dataset_file ./datasets/piper_dataset.hdf5 --num_demos 10`

**2. IK-Rel回放数据**
- `./isaaclab.sh -p scripts/tools/replay_demos.py --task Isaac-Piper-Grab-IK-Rel-v0 --device cpu --dataset_file ./datasets/piper_dataset.hdf5`

**3. IK-Rel标注子任务**
- `./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py --device cpu --task Isaac-Piper-Grab-IK-Rel-Mimic-v0 --input_file ./datasets/piper_dataset.hdf5 --output_file ./datasets/annotated_piper_dataset.hdf5`

**4. IK-Rel数据生成**
- `./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py --device cpu --generation_num_trials 10 --input_file ./datasets/annotated_piper_dataset.hdf5 --output_file ./datasets/generated_dataset_small_piper_grab.hdf5`

`grab_joint_pos_env_cfg`中物体随机化的范围缩小后，生成数据成功率上升

-----------------

**1. Visuo记录数据**
- `./isaaclab.sh -p scripts/tools/record_demos.py --task Isaac-Piper-Grab-IK-Rel-Visuomotor-Mimic-v0 --device cpu --teleop_device keyboard --dataset_file ./datasets/visuo_dataset.hdf5 --num_demos 10 --enable_cameras`

**2. Visuo标注数据**
`./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py  --device cpu --task Isaac-Piper-Grab-IK-Rel-Visuomotor-Mimic-v0 --auto --input_file ./datasets/visuo_dataset.hdf5 --output_file ./datasets/visuo_annotated_dataset.hdf5 --enable_cameras`

----------------
**1. Cosmos记录数据**
- `./isaaclab.sh -p scripts/tools/record_demos.py --task Isaac-Piper-Grab-IK-Rel-Visuomotor-Cosmos-Mimic-v0 --device cpu --teleop_device keyboard --dataset_file ./datasets/cosmos_dataset.hdf5 --num_demos 10 --enable_cameras`
**2. Cosmos标注数据**
- `./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py --device cpu --task Isaac-Piper-Grab-IK-Rel-Visuomotor-Cosmos-Mimic-v0 --input_file ./datasets/cosmos_dataset.hdf5 --output_file ./datasets/cosmos_annotated_dataset.hdf5`