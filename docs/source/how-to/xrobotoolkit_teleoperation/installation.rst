.. _xrobotoolkit-teleoperation-installation:

环境准备
========

目标
----

让 Isaac Lab 的 Python 环境能够导入 XRoboToolkit 遥操客户端，同时避免把
XRoboToolkit 仓库的完整依赖集合解析进 Isaac Lab 环境。

最小安装
--------

在 Isaac Lab 仓库根目录执行：

.. code:: bash

   ./isaaclab.sh -p -m pip install --no-deps -e /home/kongqingwei/XRoboToolkit-Teleop-Sample-Python

如果运行时提示缺少 ``xrobotoolkit_sdk``，只把 XRoboToolkit PC Service Pybind 包安装到
Isaac Lab Python 环境中。不要优先使用 XRoboToolkit 仓库的完整环境安装脚本覆盖 Isaac Lab 环境。

快速检查
--------

.. code:: bash

   ./isaaclab.sh -p - <<'PY'
   from isaaclab.devices.xrobotoolkit import XRoboToolkitDeviceCfg
   print(XRoboToolkitDeviceCfg())
   PY

预期结果：命令能够导入 ``XRoboToolkitDeviceCfg`` 并打印配置对象。

Pixi 任务
---------

如果使用当前仓库的 Pixi 安装流，XRoboToolkit 应作为显式任务安装，而不是隐藏在
``install-isaaclab`` 的内部命令中。可先检查任务图：

.. code:: bash

   pixi task list

已知边界
--------

* ``--no-deps -e`` 只负责让 Isaac Lab Python 能 import XRoboToolkit 源码。
* XRoboToolkit PC 服务、头显端应用、控制器输入、网络连通性不由上述命令启动。
* 若 Isaac Lab 在 Docker 容器中运行，容器内也需要能访问本地 XRoboToolkit 源码路径。

