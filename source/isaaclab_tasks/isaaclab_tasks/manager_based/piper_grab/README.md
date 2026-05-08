## 仿真数据生成——任务规划
### 1.基础环境
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

目前的piper任务空间抓取(无视觉)可以通过遥操作记录演示数据并回放演示数据，但还无法标注子任务，需要配置skillgen环境。

### 2.Skillgen数据采集、标注与生成
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
