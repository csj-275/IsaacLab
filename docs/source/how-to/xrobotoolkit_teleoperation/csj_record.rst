

启动容器
--------
.. code:: bash
  ./docker/container.py enter base


遥操作
-------
.. code:: bash
  ./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
  --task Isaac-Piper-Grab-IK-Rel-Mimic-v0 \
  --teleop_device xrobotoolkit \
  --xrobotoolkit_control_mode absolute


记录演示
-------
.. code::bash
  ./isaaclab.sh -p scripts/tools/record_demos.py \
  --task Isaac-Piper-Grab-IK-Rel-Mimic-v0 \
  --teleop_device xrobotoolkit \
  --dataset_file ./datasets/piper_dataset.hdf5 \
  --num_demos 10 \