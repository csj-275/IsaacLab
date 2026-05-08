.. _xrobotoolkit-teleoperation:

XRoboToolkit Teleoperation
==========================

本节记录 Isaac Lab 中 XRoboToolkit 遥操的本地使用方式。文档按模块组织：
先确认依赖与任务配置，再运行交互遥操或 HDF5 录制，需要时开启腕部相机视频流与坐标映射诊断。

适用范围
--------

该文档面向当前 Isaac Lab checkout 中的 XRoboToolkit 设备集成：

* 遥操设备实现：``isaaclab.devices.xrobotoolkit.XRoboToolkitDevice``。
* 交互入口：``scripts/environments/teleoperation/teleop_se3_agent.py``。
* 演示录制入口：``scripts/tools/record_demos.py``。
* 当前 Piper 示例任务：``Isaac-Stack-Cube-Piper-IK-Rel-v0``。

.. warning::

   XRoboToolkit 依赖外部本地仓库与 PC 服务。本文只描述 Isaac Lab 侧的接口和运行命令；
   真机、头显、网络和 XRoboToolkit 服务状态需要按实际设备环境确认。

模块索引
--------

.. toctree::
   :maxdepth: 1

   installation
   task_contract
   teleop
   recording
   video_stream
   calibration
   troubleshooting

