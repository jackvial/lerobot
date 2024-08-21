#!/bin/bash
REPO_ID="jackvial/koch_pick_and_place_pistachio_11_e20"
python lerobot/scripts/control_robot.py record  --fps 15 --root tmp/data --repo-id $REPO_ID  --warmup-time-s 4 --episode-time-s 8  --reset-time-s 8  --num-episodes 20  --run-compute-stats 1 --robot-path lerobot/configs/robot/koch_jack.yaml