.. _xrobotoolkit-teleoperation-calibration:

坐标映射与标定
==============

默认策略
--------

XRoboToolkit 设备默认使用 ``mapping_mode="world_frame_calibrated"``。该模式优先读取
``--xrobotoolkit_calibration_json`` 指向的标定 JSON：

* 平移 delta 使用 ``W_T_Q[:3,:3]`` 映射到 Isaac Lab / ROS world 语义。
* 旋转 delta 先使用 ``W_T_Q[:3,:3]`` 做轴映射，再使用 ``R_rot_map`` 做方向映射。

如果未提供标定 JSON，设备不会中止运行；它会打印一次 uncalibrated fallback 提示，并使用
内置 OpenXR-to-ROS 轴映射。该 fallback 只用于兼容和诊断，不代表已经完成 world-frame 标定。

IsaacLab 可视化校准
-------------------

推荐使用 Isaac Sim Viewport 原生校准脚本生成 JSON：

.. code:: bash

   TERM=xterm ./isaaclab.sh -p scripts/tools/calibrate_xrobotoolkit_visual.py

该脚本构建 Isaac Lab Piper 场景，使用 ``FrameTransformer`` 获取当前 TCP 位姿，并在 Viewport
中显示 Piper 当前 TCP、TCP 锚点、6 个世界系目标点、XR controller 原始/ROS/标定后 frame、
以及拟合后的 TCP target frame。脚本不连接 Piper SDK，不发送 CAN、``move_j`` 或夹爪命令。

交互流程：

1. 将控制器放到舒适初始姿态，按 ``right_grip`` 预锚定。
2. 依次把控制器中心移动到 TCP 原点、``+/-X 0.12 m``、``+/-Y 0.12 m``、``+Z 0.10 m``，
   每个位置按 ``A`` 采样；按 ``B`` 重做当前样本；按 ``right_axis_click`` 退出。
3. 平移质量通过后，依次完成 roll ``+X``、pitch ``-Y``、yaw ``+Z`` 三个方向型旋转动作。
4. 脚本输出 JSON 到 ``logs/piper_calibration/``，并打印可直接用于交互遥操和录制的命令。

默认质量门槛：

* 平移：``mean_position_error_m <= 0.03 m``，``max_position_error_m <= 0.08 m``。
* 旋转动作角度：``15 deg <= angle <= 150 deg``。
* 旋转方向：``mean_axis_error_deg <= 25 deg``，``max_axis_error_deg <= 40 deg``。

使用通过质量门槛的 JSON：

.. code:: bash

   TERM=xterm ./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
       --task Isaac-Stack-Cube-Piper-IK-Rel-v0 \
       --teleop_device xrobotoolkit \
       --xrobotoolkit_mapping_mode world_frame_calibrated \
       --xrobotoolkit_calibration_json logs/piper_calibration/xrobotoolkit_world_frame_calibration_YYYYMMDD_HHMMSS.json

开启映射诊断
------------

交互遥操：

.. code:: bash

   TERM=xterm ./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
       --task Isaac-Stack-Cube-Piper-IK-Rel-v0 \
       --teleop_device xrobotoolkit \
       --xrobotoolkit_debug_mapping

录制演示：

.. code:: bash

   TERM=xterm ./isaaclab.sh -p scripts/tools/record_demos.py \
       --headless \
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

fallback 映射
-------------

未提供标定 JSON 时，fallback 位置与旋转映射均为：

.. code:: text

   robot_x = -raw_z
   robot_y = -raw_x
   robot_z =  raw_y

对应配置字段：

* ``XRoboToolkitDeviceCfg.delta_pos_axis_map``
* ``XRoboToolkitDeviceCfg.delta_rot_axis_map``

这些字段仍保留为 ``mapping_mode="axis_map"`` 的诊断入口。默认路径应优先使用
``calibrate_xrobotoolkit_visual.py`` 生成 JSON，而不是直接修改高层遥操脚本的数据流。

.. _xrobotoolkit-teleoperation-calibration-comparison:

与 Piper 硬件遥操方案对比
--------------------------

``external/xrobotoolkit/xrobotoolkit`` submodule 中提供了完整的 piper 硬件遥操参考实现，
其映射和标定思路与当前 Isaac Lab 集成存在显著差异。

映射方式：delta vector vs 4x4 齐次矩阵
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**当前 Isaac Lab 实现 —— 平移旋转完全分离**

设备使用 **delta vector + 独立 3x3 轴映射矩阵**。位置和旋转的 delta 各自独立计算、
各自通过不同的 3x3 矩阵映射，二者不存在耦合：

.. code:: python

    # source/isaaclab/isaaclab/devices/xrobotoolkit/xrobotoolkit_device.py
    raw_delta_pos = (pos - self._ref_pos) * self.pos_sensitivity
    delta_quat = _quat_multiply(quat, _quat_conjugate(self._ref_quat))
    raw_delta_rot = _quat_to_rotvec(delta_quat) * self.rot_sensitivity

    delta_pos = self._delta_pos_axis_map @ raw_delta_pos   # 位置走 3x3 映射
    delta_rot = self._delta_rot_axis_map @ raw_delta_rot   # 旋转走 3x3 映射

输出为 ``[dx, dy, dz, rx, ry, rz, gripper]``（relative 模式）或
``[x, y, z, qw, qx, qy, qz, gripper]``（absolute 模式）。

**Sample 实现 —— 4x4 齐次矩阵框架，但平移旋转分路径计算**

Sample 使用 4x4 SE(3) 矩阵作为统一框架，所有 delta 模式最终输出完整的 4x4 ``target_T``，
由 Placo FrameTask 驱动全 6-DoF IK。其 ``world_frame_calibrated`` 模式的核心逻辑
（``xrobotoolkit_teleop/hardware/piper_world_frame_mapping.py``）：

.. code:: python

    # 用标定好的 W_T_Q 将手柄位姿映射到 ROS 世界系
    W_T_H0 = W_T_Q @ Q_T_H0        # 参考手柄在世界系 (4x4)
    W_T_Ht = W_T_Q @ Q_T_Ht        # 当前手柄在世界系 (4x4)

    # 平移：世界系下纯位置差，线性缩放
    target[:3, 3] = W_T_TCP0[:3, 3] + scale * (W_T_Ht[:3, 3] - W_T_H0[:3, 3])

    # 旋转：SO(3) delta 经 R_rot_map 方向映射后叠加到 TCP 参考姿态
    R_delta_controller = project_to_so3(W_T_Ht[:3,:3] @ W_T_H0[:3,:3].T)
    R_delta_target = apply_rotation_direction_map(R_delta_controller, R_rot_map)
    target[:3,:3] = project_to_so3(R_delta_target @ W_T_TCP0[:3,:3])

Sample 的平移和旋转在 4x4 框架内部走的也是不同的计算路径（平移线性缩放、旋转 SO(3) log/exp），
但它们共享同一个空间映射 ``W_T_Q``，最终合并为统一的 4x4 target。

.. list-table:: 映射方式差异总结
   :header-rows: 1

   * - 维度
     - Sample
     - IsaacLab
   * - 数学框架
     - 4x4 齐次矩阵
     - 3-vector delta
   * - 空间映射
     - 单个 ``W_T_Q`` (4x4) 统一映射
     - 标定 JSON 中的 ``W_T_Q[:3,:3]``；无 JSON 时回退到两个 3x3 fallback 矩阵
   * - 映射来源
     - 标定（SVD 拟合）
     - IsaacLab 可视化标定 JSON；fallback 为硬编码常量
   * - 旋转处理
     - SO(3) delta + ``R_rot_map`` 方向映射
     - rotation vector + ``W_T_Q[:3,:3]`` + ``R_rot_map``；fallback 为 3x3 axis map
   * - 输出形式
     - 4x4 target_T → Placo FrameTask
     - (dx,dy,dz,rx,ry,rz) 或绝对位姿
   * - 耦合程度
     - 共享 W_T_Q 框架，分路径计算
     - 共享 W_T_Q 的旋转部分，但仍输出 IsaacLab delta/absolute 命令

.. _xrobotoolkit-teleoperation-calibration-migration:

标定思路迁移
------------

Sample 的三阶段标定
^^^^^^^^^^^^^^^^^^^

Sample 提供了完整的自动化标定管线（``scripts/hardware/calibrate_piper_vr_controller_frame_visual.py``）：

1. **单点锚定**：将手柄放在 TCP 位置，用 ``openxr_anchor_W_T_Q()`` 计算初始 ``W_T_Q``
   （旋转固定为 OPENXR_TO_ROS，平移由位置差确定）。

2. **多点 SVD 标定**：在 6 个预设世界坐标点（TCP 原点、±X 0.12m、±Y 0.12m、+Z 0.10m）
   用手柄采样，通过 Kabsch/SVD 算法估计最优 ``W_T_Q``：

   .. code:: python

       # H = quest_centered.T @ world_centered
       # U, S, Vt = svd(H)
       # R = Vt.T @ U.T (含 det>0 修正)
       # t = world_centroid - R @ quest_centroid

3. **旋转方向标定**：用户演示 roll/pitch/yaw 三个旋转动作，通过 SVD 拟合 ``R_rot_map``，
   将手柄旋转轴映射到期望的 TCP 旋转轴。

4. **质量检查**：平移 mean_error < 0.03m, max < 0.08m；旋转轴误差 mean < 25°, max < 40°。
   通过后输出 JSON（含 ``W_T_Q``, ``R_align``, ``R_rot_map``）。

IsaacLab 当前实现
^^^^^^^^^^^^^^^^^

IsaacLab 已复用 Sample 中的标定数学模块（``piper_world_frame_mapping.py``）：

* ``estimate_W_T_Q()`` —— SVD 估计 OpenXR→World 的 3x3 旋转矩阵
* ``estimate_rotation_direction_map()`` —— SVD 估计旋转方向映射
* ``project_to_so3()``, ``so3_log()``, ``so3_exp()`` —— SO(3) 工具函数
* ``load_piper_world_calibration_json()`` —— 标定 JSON 加载
* 质量检查逻辑（reprojection error, rotation axis error）

当前实现保留了 IsaacLab 的 delta vector / absolute pose 输出接口，因此仍有以下差异：

1. **W_T_Q 的平移分量对 delta-from-reference 无效**：Isaac Lab 使用相对于激活参考的
   delta 机制，平移的绝对原点会在差分化中消去。只有 ``W_T_Q`` 的 3x3 旋转分量
   （即 ``W_T_Q[:3, :3]``）对 Isaac Lab 有意义——它等价于 ``delta_pos_axis_map``
   和 ``delta_rot_axis_map`` 的功能。

2. **R_rot_map 作为独立环节**：``world_frame_calibrated`` 模式中，旋转先由
   ``W_T_Q[:3,:3]`` 进入 world 语义，再经 ``R_rot_map`` 做方向映射。``axis_map``
   模式保留旧的 3x3 rotation-vector 映射。

3. **标定脚本适配 Isaac Sim 环境**：Sample 的可视化脚本依赖 Placo/MeshCat 获取和显示
   TCP；IsaacLab 使用 ``InteractiveScene``、Piper asset 和 ``FrameTransformer`` 获取 TCP，
   并用 Isaac Sim Viewport markers 显示目标点和 frame。
