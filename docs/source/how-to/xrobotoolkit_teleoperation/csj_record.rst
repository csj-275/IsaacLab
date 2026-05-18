

启动容器
--------
.. code:: bash
  ./docker/container.py start base --files docker-compose.xrobotoolkit.patch.yaml
  ./docker/container.py enter base

.. code:: bash
  TERM=xterm ./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py --task Isaac-Stack-Cube-Piper-IK-Rel-v0 --teleop_device xrobotoolkit --xrobotoolkit_mapping_mode world_frame_calibrated --xrobotoolkit_calibration_json logs/piper_calibration/xrobotoolkit_world_frame_calibration_20260515_095649.json

遥操作
-------
.. code:: bash
  IK
  -----
  TERM=xterm ./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
  --task Isaac-Piper-Grab-IK-Rel-Mimic-v0 \
  --teleop_device xrobotoolkit \
  --xrobotoolkit_mapping_mode world_frame_calibrated \
  --xrobotoolkit_calibration_json logs/piper_calibration/xrobotoolkit_world_frame_calibration_20260515_095649.json

  
记录演示
-------
.. code::bash
  Cosmos
  ------
  TERM=xterm ./isaaclab.sh -p scripts/tools/record_demos.py \
       --task Isaac-Piper-Grab-IK-Rel-Visuomotor-Mimic-v0 \
       --teleop_device xrobotoolkit \
       --xrobotoolkit_mapping_mode world_frame_calibrated \
       --xrobotoolkit_calibration_json logs/piper_calibration/xrobotoolkit_world_frame_calibration_20260515_095649.json\
       --dataset_file ./datasets/visuo_xrobo_dataset.hdf5 \
       --num_demos 10 \
       --enable_cameras
