#!/bin/bash

python lerobot/scripts/control_robot.py record  --fps 15 --root tmp/data --repo-id jackvial/koch_pick_and_place_pistachio_8_e100  --warmup-time-s 4 --episode-time-s 8  --reset-time-s 8  --num-episodes 100  --run-compute-stats 1 --robot-path lerobot/configs/robot/koch_jack.yaml