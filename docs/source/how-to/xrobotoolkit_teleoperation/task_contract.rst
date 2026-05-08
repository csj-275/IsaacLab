.. _xrobotoolkit-teleoperation-task-contract:

任务配置约定
============

XRoboToolkit 设备
-----------------

Isaac Lab 侧设备配置为 ``XRoboToolkitDeviceCfg``，当前默认输入约定如下：

.. list-table::
   :header-rows: 1

   * - 字段
     - 默认值
     - 语义
   * - ``pose_source``
     - ``right_controller``
     - 读取右手控制器位姿。
   * - ``control_trigger``
     - ``right_grip``
     - 按住后进入遥操激活状态。
   * - ``gripper_trigger``
     - ``right_trigger``
     - 控制夹爪开合。
   * - ``reset_button``
     - ``A``
     - 触发 Isaac Lab 环境 reset 回调。
   * - ``control_mode``
     - ``absolute``
     - 输出绝对末端目标位姿；也支持 ``relative``。

动作维度
--------

``XRoboToolkitDevice`` 根据控制模式输出不同动作：

* ``relative``：``[dx, dy, dz, rx, ry, rz, gripper]``。
* ``absolute``：``[x, y, z, qw, qx, qy, qz, gripper]``。

绝对模式要求任务配置提供 ``env_cfg.scene.ee_frame``，运行入口会把当前末端执行器位姿作为
激活参考。如果任务没有 ``ee_frame``，绝对模式会报错。

Piper 示例任务
--------------

当前 Piper 任务注册名：

.. code:: text

   Isaac-Stack-Cube-Piper-IK-Rel-v0

任务配置位于：

.. code:: text

   source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack/config/piper/stack_ik_rel_env_cfg.py

机器人资产配置位于：

.. code:: text

   source/isaaclab_assets/isaaclab_assets/robots/piper.py

Piper URDF 和 mesh 资产位于：

.. code:: text

   source/isaaclab_assets/isaaclab_assets/data/piper

该任务已经配置：

* Piper 机械臂与夹爪 action。
* ``ee_frame``，用于绝对模式末端位姿参考。
* ``wrist_cam``，用于 XRoboToolkit 视频流。
* ``teleop_devices.devices["xrobotoolkit"]``，默认 ``control_mode="absolute"``。

坐标基准
--------

当前 XRoboToolkit 默认位置映射面向 ROS 机器人基座语义：``X`` 前、``Y`` 左、``Z`` 上。
默认映射把 XR SDK 输入增量映射为：

.. code:: text

   robot = [-raw_z, -raw_x, raw_y]

如果实际机器人效果与该语义不一致，应先使用映射诊断日志确认输入、映射和机器人效果，
不要直接修改任务 action 或控制器名称。
