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
   * - ``mapping_mode``
     - ``world_frame_calibrated``
     - 默认使用 world-frame 标定策略；未提供标定 JSON 时回退到当前轴映射。

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
* ``teleop_devices.devices["xrobotoolkit"]``，默认 ``mapping_mode="world_frame_calibrated"``。

坐标基准
--------

当前 XRoboToolkit 默认映射策略为 ``world_frame_calibrated``。如果运行时提供
``calibration_json``，absolute 模式会在按下 ``right_grip`` 时捕获控制器参考位姿和
当前 TCP 参考位姿，随后使用完整 ``W_T_Q`` 计算世界系平移 delta，并用 SO(3) delta +
``R_rot_map`` 计算旋转 delta，最终输出绝对 TCP 目标位姿。relative 模式仍输出 6D
delta-vector，使用 ``W_T_Q[:3,:3]`` 映射平移/旋转轴，并用 ``R_rot_map`` 继续映射旋转方向。
如果未提供 JSON，设备会打印一次 uncalibrated fallback 提示，并回退到内置 OpenXR-to-ROS
轴映射。

fallback 映射面向 ROS 机器人基座语义：``X`` 前、``Y`` 左、``Z`` 上。
它把 XR SDK 输入增量映射为：

.. code:: text

   robot = [-raw_z, -raw_x, raw_y]

如果实际机器人效果与该语义不一致，应先运行可视化标定或使用映射诊断日志确认输入、
映射和机器人效果，不要直接修改任务 action 或控制器名称。
