.. _xrobotoolkit-teleoperation-video-stream:

腕部相机视频流
==============

入口行为
--------

``teleop_se3_agent.py`` 使用 ``--teleop_device xrobotoolkit`` 时，默认尝试启动
``XRoboToolkitVideoStreamServer``：

* 默认控制监听地址：``0.0.0.0:13579``。
* 默认 scene camera：``wrist_cam``。
* 默认输出尺寸：``640x480`` 像素。
* 默认帧率：``30`` FPS。
* 默认码率：``4000000`` bit/s。
* 默认只接受 camera type 为 ``ZED`` 的 ``OPEN_CAMERA`` 请求。

常用命令
--------

指定监听地址和相机：

.. code:: bash

   TERM=xterm ./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
       --headless \
       --task Isaac-Stack-Cube-Piper-IK-Rel-v0 \
       --teleop_device xrobotoolkit \
       --xrobotoolkit_video_listen 0.0.0.0:13579 \
       --xrobotoolkit_video_camera wrist_cam

调整视频参数：

.. code:: bash

   TERM=xterm ./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
       --headless \
       --task Isaac-Stack-Cube-Piper-IK-Rel-v0 \
       --teleop_device xrobotoolkit \
       --xrobotoolkit_video_width 640 \
       --xrobotoolkit_video_height 480 \
       --xrobotoolkit_video_fps 30 \
       --xrobotoolkit_video_bitrate 4000000

协议边界
--------

Isaac Lab 侧视频服务处理 XRoboToolkit 的 ``OPEN_CAMERA`` / ``CLOSE_CAMERA`` 控制流。
视频帧来自 Isaac Lab scene camera 的 RGB 输出，编码后发送给请求端。深度或多相机协议未在当前
Piper 示例路径中作为默认能力声明。

Piper 相机配置
--------------

Piper 示例任务的 ``wrist_cam`` 挂在 ``camera_link`` 下，使用 RealSense 光学坐标转换：

.. code:: text

   rot=(0.5, -0.5, 0.5, -0.5), convention="ros"

如果画面全白、方向异常或没有视频，先分别排查 scene camera 内容和 XRoboToolkit 传输链路。
不要把二者合并成同一个故障结论。
