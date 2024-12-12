python lerobot/scripts/control_robot.py record \
    --robot-path lerobot/configs/robot/koch_jack.yaml \
    --fps 30 \
    --root outputs/koch_with_rewards_4 \
    --repo-id jackvial/koch_with_rewards_4 \
    --warmup-time-s 2 \
    --episode-time-s 20 \
    --reset-time-s 5 \
    --num-episodes 10 \
    --push-to-hub 1 \
    --assign-rewards 1 \
    --single-task test_description

