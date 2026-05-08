.. _xrobotoolkit-teleoperation-calibration:

坐标映射与标定
==============

开启映射诊断
------------

交互遥操：

.. code:: bash

   ./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
       --task Isaac-Stack-Cube-Piper-IK-Rel-v0 \
       --teleop_device xrobotoolkit \
       --xrobotoolkit_debug_mapping

录制演示：

.. code:: bash

   ./isaaclab.sh -p scripts/tools/record_demos.py \
       --task Isaac-Stack-Cube-Piper-IK-Rel-v0 \
       --teleop_device xrobotoolkit \
       --xrobotoolkit_debug_mapping

日志格式
--------

启用后会周期性打印：

.. code:: text

   [XRoboToolkit mapping] mode=... raw_pos=... mapped_pos=... raw_rot=... mapped_rot=...

其中：

* ``raw_pos`` / ``raw_rot``：XR SDK 输入相对激活参考的原始增量。
* ``mapped_pos`` / ``mapped_rot``：经过 Isaac Lab 侧映射矩阵后的机器人控制增量。
* ``mode``：当前 ``relative`` 或 ``absolute`` 控制模式。

标定原则
--------

* 只在按住 ``right_grip`` 且机器人实际接受遥操命令时采集映射日志。
* 先单轴移动控制器，确认 ``raw_*`` 与 ``mapped_*`` 的符号和轴向。
* 以 ROS 基座语义为目标：``X`` 前、``Y`` 左、``Z`` 上。
* 如果真实机器人效果与日志不一致，先确认 IK、末端 frame、任务 action 和硬件反馈，再修改映射。

默认映射
--------

默认位置与旋转映射均为：

.. code:: text

   robot_x = -raw_z
   robot_y = -raw_x
   robot_z =  raw_y

对应配置字段：

* ``XRoboToolkitDeviceCfg.delta_pos_axis_map``
* ``XRoboToolkitDeviceCfg.delta_rot_axis_map``

这些字段是低影响标定入口。优先修改设备配置或任务中的 ``XRoboToolkitDeviceCfg``，
不要先改高层遥操脚本的数据流。

