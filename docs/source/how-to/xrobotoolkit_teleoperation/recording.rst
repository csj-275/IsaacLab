.. _xrobotoolkit-teleoperation-recording:

录制演示数据
============

基本命令
--------

使用 XRoboToolkit 录制 HDF5 演示：

.. code:: bash

   TERM=xterm ./isaaclab.sh -p scripts/tools/record_demos.py \
       --headless \
       --task Isaac-Stack-Cube-Piper-IK-Rel-v0 \
       --teleop_device xrobotoolkit \
       --dataset_file ./datasets/piper_xrobo_demo.hdf5 \
       --num_demos 10

常用参数
--------

.. list-table::
   :header-rows: 1

   * - 参数
     - 默认值
     - 说明
   * - ``--dataset_file``
     - ``./datasets/dataset.hdf5``
     - HDF5 输出路径。
   * - ``--step_hz``
     - ``30``
     - 环境步进频率，单位 Hz。
   * - ``--num_demos``
     - ``0``
     - 目标演示条数；``0`` 表示不设上限。
   * - ``--num_success_steps``
     - ``10``
     - 连续成功步数达到该值后结束当前 demo。
   * - ``--xrobotoolkit_control_mode``
     - 任务配置值
     - 覆盖 XRoboToolkit 控制模式。
   * - ``--xrobotoolkit_mapping_mode``
     - 任务配置值
     - 覆盖 XRoboToolkit 映射模式，支持 ``world_frame_calibrated`` 和 ``axis_map``。
   * - ``--xrobotoolkit_calibration_json``
     - ``None``
     - 使用可视化标定脚本生成的 world-frame calibration JSON。
   * - ``--xrobotoolkit_debug_mapping``
     - ``False``
     - 打印原始与映射后的控制器增量。

使用标定 JSON
-------------

推荐先运行可视化校准脚本，使用通过质量门槛的 JSON 录制：

.. code:: bash

   TERM=xterm ./isaaclab.sh -p scripts/tools/record_demos.py \
       --headless \
       --task Isaac-Stack-Cube-Piper-IK-Rel-v0 \
       --teleop_device xrobotoolkit \
       --xrobotoolkit_mapping_mode world_frame_calibrated \
       --xrobotoolkit_calibration_json logs/piper_calibration/xrobotoolkit_world_frame_calibration_YYYYMMDD_HHMMSS.json \
       --dataset_file ./datasets/piper_xrobo_demo.hdf5 \
       --num_demos 10

未提供 JSON 时会回退到内置 OpenXR-to-ROS 轴映射，并打印 uncalibrated fallback 提示。

记录边界
--------

``record_demos.py`` 记录的是 Isaac Lab 环境中的演示轨迹，不证明真实机器人部署成功。
如果要验证真机行为，需要在硬件遥操路径上独立检查关节、末端、夹爪和安全 reset 逻辑。
