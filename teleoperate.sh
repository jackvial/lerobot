#/bin/bash

python lerobot/scripts/control_robot.py teleoperate --robot-path lerobot/configs/robot/koch_bimanual_jack.yaml --robot-overrides '~cameras' --fps 30
# python lerobot/scripts/control_robot.py teleoperate --robot-path lerobot/configs/robot/koch_jack.yaml --robot-overrides '~cameras' --fps 30