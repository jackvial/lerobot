#!/bin/bash

# TODO - configure the gemini API key
python lerobot/scripts/control_robot.py \
    --config_path lerobot/configs/robot/koch_jack.yaml \
    --control.type=execute_policy \
    --control.policy.type=gemini \
    --control.policy.prompt="Put nuts in bowl" \
    --control.policy.action_dim=6 \
    --control.policy.n_action_steps=10 \
    --control.policy.use_amp=false \
    --control.policy.device=cpu \
    --control.num_episodes=1 \
    --control.episode_time_s=30 \
    --control.fps=15 \
    --control.display_data=false \
    --control.play_sounds=true
    # Add any other robot-specific or top-level arguments if needed e.g. --device=cuda for robot perception if separate