## 仿真数据生成——任务规划
1. piper关节空间控制: `piper_env_cfg` - 完成
2. piper任务空间控制: `piper_ik_env_cfg` - 完成
3. grab任务的mdp观测与终止条件: `mdp.observations`, `mdp.terminations` - 完成
4. grab环境: `grab_env_cfg` - 完成 
5. grab环境随机化: `grab_instance_randomize_env_cfg` - 完成  
6. piper关节空间的grab环境： `grab_joint_pos_env_cfg` - 初步完成测试 - offset need to check
7. piper关节空间的grab环境随机化： `grab_joint_pos_instance_randomize_env_cfg` - 初步完成测试 
8. piper任务空间的grab环境： `grab_ik_env_cfg` - 进行中
9. piper任务空间的grab环境随机化 - 待做
10. piper抓取随机物体演示 - 待做
11. piper抓取随机物体相机添加 - 待做
12. piper抓取数据集制作 - 待做
- 参考`scripts/tools/record_demos.py`
13. 数据标注
- 参考 `scripts/imitation_learning/isaaclab_mimic/annotate_demos.py`
14. 数据生成
- 参考 `scripts/imitation_learning/isaaclab_mimic/generate_dataset.py`
15. 数据评估

**模块关系**
- grab环境包含与机器人构型无关的抓取环境配置，但与待抓取物体个数、类型有关
- 具体机器人继承grab环境，配置robot和ee_frame以及待抓取物体，先关节空间，再任务空间；确定环境和随机环境无继承关系，通常任务空间环境继承自关节空间环境