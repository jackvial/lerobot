from lerobot.common.policies.act.modeling_act import ACTPolicy
import time
import torch
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.utils.utils import init_hydra_config

inference_time_s = 10
fps = 15
device = "cuda"  # TODO: On Mac, use "mps" or "cpu"

job_name="act_koch_pick_and_place_pistachio_8_e100_2_and_10_e20_and_11_e20_004"
ckpt_path = "outputs/train/act_koch_pick_and_place_pistachio_8_e100_2_and_10_e20_and_11_e20_004/checkpoints/last/pretrained_model"
robot_path = "lerobot/configs/robot/koch_jack.yaml"
policy = ACTPolicy.from_pretrained(ckpt_path)
policy.to(device)

robot_cfg = init_hydra_config(robot_path)
robot = make_robot(robot_cfg)
robot.connect()

def busy_wait(seconds):
    # Significantly more accurate than `time.sleep`, and mendatory for our use case,
    # but it consumes CPU cycles.
    # TODO(rcadene): find an alternative: from python 11, time.sleep is precise
    end_time = time.perf_counter() + seconds
    while time.perf_counter() < end_time:
        pass

for _ in range(inference_time_s * fps):
    start_time = time.perf_counter()

    # Read the follower state and access the frames from the cameras
    observation = robot.capture_observation()

    # Convert to pytorch format: channel first and float32 in [0,1]
    # with batch dimension
    for name in observation:
        if "image" in name:
            observation[name] = observation[name].type(torch.float32) / 255
            observation[name] = observation[name].permute(2, 0, 1).contiguous()
        observation[name] = observation[name].unsqueeze(0)
        observation[name] = observation[name].to(device)

    # Compute the next action with the policy
    # based on the current observation
    action = policy.select_action(observation)
    # Remove batch dimension
    action = action.squeeze(0)
    # Move to cpu, if not already the case
    action = action.to("cpu")
    # Order the robot to move
    robot.send_action(action)

    dt_s = time.perf_counter() - start_time
    busy_wait(1 / fps - dt_s)