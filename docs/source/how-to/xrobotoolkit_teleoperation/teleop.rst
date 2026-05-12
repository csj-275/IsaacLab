.. _xrobotoolkit-teleoperation-run:

运行交互遥操
============

基本命令
--------

在 Isaac Lab 仓库根目录运行 Piper 示例任务：

.. code:: bash

   TERM=xterm ./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
       --headless \
       --task Isaac-Stack-Cube-Piper-IK-Rel-v0 \
       --teleop_device xrobotoolkit

Docker 容器内默认使用 ``--headless``。如果需要 Isaac Sim 可视化窗口，必须从已经能访问本机
X Server 的宿主机终端启动或重新进入容器：

.. code:: bash

   ./docker/container.py start base --files docker-compose.xrobotoolkit.patch.yaml
   ./docker/container.py enter base

``container.py enter`` 会刷新容器使用的 xauth 文件，并把 ``localhost:10.0`` 这类宿主机
``DISPLAY`` 规范为容器内更稳定的 ``:10`` Unix socket 形式。进入容器后可检查：

.. code:: bash

   printf 'DISPLAY=%s\nXAUTHORITY=%s\n' "$DISPLAY" "$XAUTHORITY"

窗口模式运行时不要传 ``--headless``：

.. code:: bash

   TERM=xterm ./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
       --task Isaac-Stack-Cube-Piper-IK-Rel-v0 \
       --teleop_device xrobotoolkit \
       --xrobotoolkit_control_mode absolute

如果仍然出现 ``Cannot setup ExternalDragDrop without a default window`` 或 GLFW 初始化错误，
先在宿主机终端确认 ``xdpyinfo`` 能打开当前 ``DISPLAY``，再退出容器并重新执行
``./docker/container.py enter base``。

交互语义
--------

* 按住 ``right_grip``：激活遥操并捕获当前 XR 控制器参考位姿。
* 松开 ``right_grip``：停止向环境施加遥操动作。
* ``right_trigger``：控制夹爪，超过阈值时输出关闭命令。
* ``A``：触发环境 reset；下一步仿真循环执行 reset 后清空遥操参考。

控制模式
--------

默认使用任务配置中的模式。可通过命令行覆盖：

.. code:: bash

   TERM=xterm ./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
       --headless \
       --task Isaac-Stack-Cube-Piper-IK-Rel-v0 \
       --teleop_device xrobotoolkit \
       --xrobotoolkit_control_mode absolute

相对模式：

.. code:: bash

   TERM=xterm ./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
       --headless \
       --task Isaac-Stack-Cube-Piper-IK-Rel-v0 \
       --teleop_device xrobotoolkit \
       --xrobotoolkit_control_mode relative

视频流开关
----------

``teleop_se3_agent.py`` 在使用 ``--teleop_device xrobotoolkit`` 且未禁用视频流时，会开启
Isaac Lab 相机并启动 XRoboToolkit 视频控制监听。若只想验证控制链路，可禁用视频流：

.. code:: bash

   TERM=xterm ./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
       --headless \
       --task Isaac-Stack-Cube-Piper-IK-Rel-v0 \
       --teleop_device xrobotoolkit \
       --disable_xrobotoolkit_video_stream

XR 模式注意事项
---------------

当前入口在 ``--xr`` 模式下会移除 scene camera configs；如果同时请求 XRoboToolkit 视频流，
入口会禁用该视频流。因此 XRoboToolkit 腕部相机视频流与 ``--xr`` 不应同时作为默认路径使用。
