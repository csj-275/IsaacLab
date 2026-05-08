.. _xrobotoolkit-teleoperation-troubleshooting:

排障
====

导入失败
--------

现象：

.. code:: text

   ModuleNotFoundError: No module named 'xrobotoolkit_teleop'

处理：

.. code:: bash

   ./isaaclab.sh -p -m pip install --no-deps -e /home/kongqingwei/XRoboToolkit-Teleop-Sample-Python

绝对模式缺少末端 frame
----------------------

现象：

.. code:: text

   XRoboToolkit absolute mode requires env_cfg.scene.ee_frame.

处理：

* 使用已配置 ``ee_frame`` 的任务，例如 ``Isaac-Stack-Cube-Piper-IK-Rel-v0``。
* 或切换到相对模式：``--xrobotoolkit_control_mode relative``。
* 若是新任务，需要在 env config 中添加 ``FrameTransformerCfg`` 作为末端位姿来源。

没有视频
--------

检查顺序：

1. 确认未传入 ``--disable_xrobotoolkit_video_stream``。
2. 确认未同时使用会移除 scene cameras 的 ``--xr`` 默认路径。
3. 确认任务 scene 中存在 ``wrist_cam`` 且类型为 ``CameraCfg``。
4. 确认 XRoboToolkit 请求连接到 ``--xrobotoolkit_video_listen`` 指定地址。
5. 若服务启动失败，检查端口 ``13579`` 是否已被占用。

画面全白或方向错误
------------------

处理：

* 先确认 Isaac Lab 中 ``wrist_cam`` 的 RGB 输出是否正常。
* Piper 当前需要 RealSense 光学坐标转换：

  .. code:: text

     rot=(0.5, -0.5, 0.5, -0.5), convention="ros"

* 如果相机内容正常但 XR 端无画面，再检查 XRoboToolkit 传输协议和网络。

方向反了
--------

处理：

* 开启 ``--xrobotoolkit_debug_mapping``。
* 单轴移动，记录 ``raw_pos``、``mapped_pos`` 和机器人实际末端效果。
* 以 ROS 基座语义 ``X`` 前、``Y`` 左、``Z`` 上判断符号。
* 优先调整 ``XRoboToolkitDeviceCfg.delta_pos_axis_map`` 或 ``delta_rot_axis_map``。

遥操激活后跳变
--------------

可能原因：

* 激活瞬间控制器位姿不稳定。
* absolute 模式下末端位姿 provider 不正确。
* reset 后没有清空遥操参考。

处理：

* 松开 ``right_grip`` 后重新按住，重新捕获参考位姿。
* 按 ``A`` 触发环境 reset。
* 检查任务 ``ee_frame`` 是否指向真实末端执行器，而不是中间 link。

