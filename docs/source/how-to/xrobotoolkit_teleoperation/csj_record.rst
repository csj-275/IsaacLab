
标定
----
.. code:: bash
  TERM=xterm ./isaaclab.sh -p scripts/tools/calibrate_xrobotoolkit_visual.py

启动容器
--------
.. code:: bash
  ./docker/container.py start base --files docker-compose.xrobotoolkit.patch.yaml
  ./docker/container.py enter base
  bash /workspace/isaaclab/scripts/tools/setup_xrobotoolkit_env.sh

退出容器
-------
.. code:: bash
  exit

.. .. code:: bash
..   TERM=xterm ./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py --task Isaac-Stack-Cube-Piper-IK-Rel-v0 --teleop_device xrobotoolkit --xrobotoolkit_mapping_mode world_frame_calibrated --xrobotoolkit_calibration_json logs/piper_calibration/xrobotoolkit_world_frame_calibration_20260515_095649.json

重启容器
------
.. code:: bash
  docker restart $(docker ps -q)

遥操作
-------
.. code:: bash
  IK
  -----
  TERM=xterm ./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
  --task Isaac-Piper-Grab-IK-Rel  -Mimic-v0 \
  --teleop_device xrobotoolkit \
  --xrobotoolkit_mapping_mode world_frame_calibrated \
  --xrobotoolkit_calibration_json logs/piper_calibration/xrobotoolkit_world_frame_calibration_20260515_095649.json

  
记录演示
-------
.. code::bash
  Cosmos
  ------
  TERM=xterm ./isaaclab.sh -p scripts/tools/record_demos.py \
       --task Isaac-Piper-Grab-IK-Rel-Visuomotor-Cosmos-Mimic-v0 \
       --teleop_device xrobotoolkit \
       --xrobotoolkit_mapping_mode world_frame_calibrated \
       --xrobotoolkit_calibration_json logs/piper_calibration/xrobotoolkit_world_frame_calibration_20260520_023544.json\
       --dataset_file ./datasets/cosmos_xrobo_dataset.hdf5 \
       --num_demos 10 \
       --xrobotoolkit_control_mode absolute \
       --enable_cameras 

标定数据
---
.. code:: bash
  xrobotoolkit_world_frame_calibration_20260515_095649.json
  xrobotoolkit_world_frame_calibration_20260519_021821.json
  xrobotoolkit_world_frame_calibration_20260520_023544.json