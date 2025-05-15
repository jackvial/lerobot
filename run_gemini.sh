#!/bin/bash

# TODO - configure the gemini API key
python lerobot/scripts/eval.py \
    --policy.type=gemini \
    --policy.prompt="Put nuts in bowl" \
    --policy.n_action_steps=10 \
    --robot.config=lerobot/configs/robot/koch_jack.yaml