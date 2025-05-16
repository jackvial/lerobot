#!/bin/bash

USER=jackvial

python lerobot/scripts/control_robot.py \
  --config_path lerobot/configs/robot/koch_jack.yaml \
  --control.type=record \
  --control.policy.path=pi0 \
  --control.single_task="Pick the orange Lego and drop it in the blue bin." \
  --control.repo_id=${USER}/eval_tmp_pi0_9 \
  --control.num_episodes=3 \
  --control.push_to_hub=false \
  --control.video=false \
  --control.fps=15