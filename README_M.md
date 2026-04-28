# 项目记录
## [A] 自定义机器人搭建流程
### 0.urdf转usd
- 打开isaacsim, file->import导入urdfmanager_based
- 在导入的机器人下的root_joint找到Articulation Root点击删除
- 在机器人的根节点，一般与urdf名称相同右键add->physics->articulation root
### 1.机器人描述
- isaaclab_assets.robots文件夹下创建机器人描述
- 在__init__.py中导入
### 2.场景配置
- 参考scripts/tutorials/02_scene中的脚本创建场景，测试机器人描述


### 3.环境配置
- ./isaaclab -n 创建内部manager_based任务
- 在创建的任务下的env_cfg中修改场景配置类，将前面通过测试的类复制过去即可
- 参考scripts/tutorials/03_envs中的脚本创建环境，主要包括ActionsCfg, ObservationsCfg, EnentCfg的配置，测试各个类的编写是否正确
- 根据通过测试的动作、观测、事件的类，完善env_cfg内容
- 注意ManagerBasedEnvCfg和ManagerBasedRLEnvCfg的修改，一个是面向传统控制，一个是面向RL，通常创建的环境是ManagerBasedRLEnvCfg，我们在create_base_env.py中一般是用ManagerBasedEnvCfg，根据自己的需求。
