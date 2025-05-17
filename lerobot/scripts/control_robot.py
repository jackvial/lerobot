# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Utilities to control a robot.

Useful to record a dataset, replay a recorded episode, run the policy on your robot
and record an evaluation dataset, and to recalibrate your robot if needed.

Examples of usage:

- Recalibrate your robot:
```bash
python lerobot/scripts/control_robot.py \
    --robot.type=so100 \
    --control.type=calibrate
```

- Unlimited teleoperation at highest frequency (~200 Hz is expected), to exit with CTRL+C:
```bash
python lerobot/scripts/control_robot.py \
    --robot.type=so100 \
    --robot.cameras='{}' \
    --control.type=teleoperate

# Add the cameras from the robot definition to visualize them:
python lerobot/scripts/control_robot.py \
    --robot.type=so100 \
    --control.type=teleoperate
```

- Unlimited teleoperation at a limited frequency of 30 Hz, to simulate data recording frequency:
```bash
python lerobot/scripts/control_robot.py \
    --robot.type=so100 \
    --control.type=teleoperate \
    --control.fps=30
```

- Record one episode in order to test replay:
```bash
python lerobot/scripts/control_robot.py \
    --robot.type=so100 \
    --control.type=record \
    --control.fps=30 \
    --control.single_task="Grasp a lego block and put it in the bin." \
    --control.repo_id=$USER/koch_test \
    --control.num_episodes=1 \
    --control.push_to_hub=True
```

- Visualize dataset:
```bash
python lerobot/scripts/visualize_dataset.py \
    --repo-id $USER/koch_test \
    --episode-index 0
```

- Replay this test episode:
```bash
python lerobot/scripts/control_robot.py replay \
    --robot.type=so100 \
    --control.type=replay \
    --control.fps=30 \
    --control.repo_id=$USER/koch_test \
    --control.episode=0
```

- Record a full dataset in order to train a policy, with 2 seconds of warmup,
30 seconds of recording for each episode, and 10 seconds to reset the environment in between episodes:
```bash
python lerobot/scripts/control_robot.py record \
    --robot.type=so100 \
    --control.type=record \
    --control.fps 30 \
    --control.repo_id=$USER/koch_pick_place_lego \
    --control.num_episodes=50 \
    --control.warmup_time_s=2 \
    --control.episode_time_s=30 \
    --control.reset_time_s=10
```

- For remote controlled robots like LeKiwi, run this script on the robot edge device (e.g. RaspBerryPi):
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=lekiwi \
  --control.type=remote_robot
```

**NOTE**: You can use your keyboard to control data recording flow.
- Tap right arrow key '->' to early exit while recording an episode and go to resseting the environment.
- Tap right arrow key '->' to early exit while resetting the environment and got to recording the next episode.
- Tap left arrow key '<-' to early exit and re-record the current episode.
- Tap escape key 'esc' to stop the data recording.
This might require a sudo permission to allow your terminal to monitor keyboard events.

**NOTE**: You can resume/continue data recording by running the same data recording command and adding `--control.resume=true`.

- Train on this dataset with the ACT policy:
```bash
python lerobot/scripts/train.py \
  --dataset.repo_id=${HF_USER}/koch_pick_place_lego \
  --policy.type=act \
  --output_dir=outputs/train/act_koch_pick_place_lego \
  --job_name=act_koch_pick_place_lego \
  --device=cuda \
  --wandb.enable=true
```

- Run the pretrained policy on the robot:
```bash
python lerobot/scripts/control_robot.py \
    --robot.type=so100 \
    --control.type=record \
    --control.fps=30 \
    --control.single_task="Grasp a lego block and put it in the bin." \
    --control.repo_id=$USER/eval_act_koch_pick_place_lego \
    --control.num_episodes=10 \
    --control.warmup_time_s=2 \
    --control.episode_time_s=30 \
    --control.reset_time_s=10 \
    --control.push_to_hub=true \
    --control.policy.path=outputs/train/act_koch_pick_place_lego/checkpoints/080000/pretrained_model
```
"""

import logging
import os
import time
from dataclasses import asdict
from pprint import pformat

import rerun as rr
import numpy as np

# from safetensors.torch import load_file, save_file
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.factory import make_policy
from lerobot.common.robot_devices.control_configs import (
    CalibrateControlConfig,
    ControlConfig,
    ControlPipelineConfig,
    ExecutePolicyControlConfig,
    RecordControlConfig,
    RemoteRobotConfig,
    ReplayControlConfig,
    TeleoperateControlConfig,
)
from lerobot.common.robot_devices.control_utils import (
    control_loop,
    init_keyboard_listener,
    is_headless,
    log_control_info,
    record_episode,
    reset_environment,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
    stop_recording,
    warmup_record,
)
from lerobot.common.robot_devices.robots.utils import Robot, make_robot_from_config
from lerobot.common.robot_devices.utils import busy_wait, safe_disconnect
from lerobot.common.utils.utils import has_method, init_logging, log_say
from lerobot.configs import parser

########################################################################################
# Control modes
########################################################################################


@safe_disconnect
def calibrate(robot: Robot, cfg: CalibrateControlConfig):
    # TODO(aliberts): move this code in robots' classes
    if robot.robot_type.startswith("stretch"):
        if not robot.is_connected:
            robot.connect()
        if not robot.is_homed():
            robot.home()
        return

    arms = robot.available_arms if cfg.arms is None else cfg.arms
    unknown_arms = [arm_id for arm_id in arms if arm_id not in robot.available_arms]
    available_arms_str = " ".join(robot.available_arms)
    unknown_arms_str = " ".join(unknown_arms)

    if arms is None or len(arms) == 0:
        raise ValueError(
            "No arm provided. Use `--arms` as argument with one or more available arms.\n"
            f"For instance, to recalibrate all arms add: `--arms {available_arms_str}`"
        )

    if len(unknown_arms) > 0:
        raise ValueError(
            f"Unknown arms provided ('{unknown_arms_str}'). Available arms are `{available_arms_str}`."
        )

    for arm_id in arms:
        arm_calib_path = robot.calibration_dir / f"{arm_id}.json"
        if arm_calib_path.exists():
            print(f"Removing '{arm_calib_path}'")
            arm_calib_path.unlink()
        else:
            print(f"Calibration file not found '{arm_calib_path}'")

    if robot.is_connected:
        robot.disconnect()

    if robot.robot_type.startswith("lekiwi") and "main_follower" in arms:
        print("Calibrating only the lekiwi follower arm 'main_follower'...")
        robot.calibrate_follower()
        return

    if robot.robot_type.startswith("lekiwi") and "main_leader" in arms:
        print("Calibrating only the lekiwi leader arm 'main_leader'...")
        robot.calibrate_leader()
        return

    # Calling `connect` automatically runs calibration
    # when the calibration file is missing
    robot.connect()
    robot.disconnect()
    print("Calibration is done! You can now teleoperate and record datasets!")


@safe_disconnect
def teleoperate(robot: Robot, cfg: TeleoperateControlConfig):
    control_loop(
        robot,
        control_time_s=cfg.teleop_time_s,
        fps=cfg.fps,
        teleoperate=True,
        display_data=cfg.display_data,
    )


@safe_disconnect
def record(
    robot: Robot,
    cfg: RecordControlConfig,
) -> LeRobotDataset:
    # TODO(rcadene): Add option to record logs
    if cfg.resume:
        dataset = LeRobotDataset(
            cfg.repo_id,
            root=cfg.root,
        )
        if len(robot.cameras) > 0:
            dataset.start_image_writer(
                num_processes=cfg.num_image_writer_processes,
                num_threads=cfg.num_image_writer_threads_per_camera * len(robot.cameras),
            )
        sanity_check_dataset_robot_compatibility(dataset, robot, cfg.fps, cfg.video)
    else:
        # Create empty dataset or load existing saved episodes
        sanity_check_dataset_name(cfg.repo_id, cfg.policy)
        dataset = LeRobotDataset.create(
            cfg.repo_id,
            cfg.fps,
            root=cfg.root,
            robot=robot,
            use_videos=cfg.video,
            image_writer_processes=cfg.num_image_writer_processes,
            image_writer_threads=cfg.num_image_writer_threads_per_camera * len(robot.cameras),
        )

    # Load pretrained policy
    # Instantiate policy. Gemini config is self-sufficient, so don't supply env/dataset.
    if cfg.policy.type == "gemini":
        policy = make_policy(cfg.policy)
    else:
        policy = make_policy(cfg.policy, ds_meta=None, env_cfg=None)

    if not robot.is_connected:
        robot.connect()

    listener, events = init_keyboard_listener()

    # Execute a few seconds without recording to:
    # 1. teleoperate the robot to move it in starting position if no policy provided,
    # 2. give times to the robot devices to connect and start synchronizing,
    # 3. place the cameras windows on screen
    enable_teleoperation = policy is None
    log_say("Warmup record", cfg.play_sounds)
    warmup_record(robot, events, enable_teleoperation, cfg.warmup_time_s, cfg.display_data, cfg.fps)

    if has_method(robot, "teleop_safety_stop"):
        robot.teleop_safety_stop()

    recorded_episodes = 0
    while True:
        if recorded_episodes >= cfg.num_episodes:
            break

        log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
        record_episode(
            robot=robot,
            dataset=dataset,
            events=events,
            episode_time_s=cfg.episode_time_s,
            display_data=cfg.display_data,
            policy=policy,
            fps=cfg.fps,
            single_task=cfg.single_task,
        )

        # Execute a few seconds without recording to give time to manually reset the environment
        # Current code logic doesn't allow to teleoperate during this time.
        # TODO(rcadene): add an option to enable teleoperation during reset
        # Skip reset for the last episode to be recorded
        if not events["stop_recording"] and (
            (recorded_episodes < cfg.num_episodes - 1) or events["rerecord_episode"]
        ):
            log_say("Reset the environment", cfg.play_sounds)
            reset_environment(robot, events, cfg.reset_time_s, cfg.fps)

        if events["rerecord_episode"]:
            log_say("Re-record episode", cfg.play_sounds)
            events["rerecord_episode"] = False
            events["exit_early"] = False
            dataset.clear_episode_buffer()
            continue

        dataset.save_episode()
        recorded_episodes += 1

        if events["stop_recording"]:
            break

    log_say("Stop recording", cfg.play_sounds, blocking=True)
    stop_recording(robot, listener, cfg.display_data)

    if cfg.push_to_hub:
        dataset.push_to_hub(tags=cfg.tags, private=cfg.private)

    log_say("Exiting", cfg.play_sounds)
    return dataset


@safe_disconnect
def execute_policy_on_robot(robot: Robot, cfg: ExecutePolicyControlConfig):
    """Runs a given policy on the robot for a specified number of episodes/duration."""
    if cfg.policy is None:
        raise ValueError(
            "A policy configuration must be provided for `execute_policy` mode. "
            "Please specify it using `--control.policy.type=<type>` and other `--control.policy.*` arguments."
        )

    # Determine action_dim and observation_spec from the robot if possible,
    # which might be needed by make_policy or the policy itself.
    # This is a placeholder for a more robust way to get this. For GeminiPolicy,
    # action_dim is taken from its own config, which should be set correctly by the user.
    # robot_attrs = robot.metadata.get("robot_attributes", {})
    # action_dim = robot_attrs.get("action_dim")
    # obs_spec = robot_attrs.get("observation_space")

    # For now, make_policy will rely on PreTrainedConfig having sufficient info (e.g., action_feature.shape)
    # or defaults. GeminiPolicy has a default action_feature shape.
    policy = make_policy(cfg.policy, ds_meta=None, env_cfg=None)
    policy.eval() # Ensure policy is in evaluation mode

    if not robot.is_connected:
        robot.connect()

    listener, events = init_keyboard_listener()

    log_say(f"Starting policy execution for {cfg.num_episodes} episode(s).", cfg.play_sounds)

    for episode_idx in range(cfg.num_episodes):
        if events.get("stop_recording", False):  # Reuse stop_recording for general stop
            log_say("Execution stopped by user.", cfg.play_sounds)
            break

        log_say(f"Starting episode {episode_idx + 1}/{cfg.num_episodes}", cfg.play_sounds)
        policy.reset()

        # A brief moment for manual preparation if needed
        if episode_idx == 0 and cfg.num_episodes > 0 : # Only for the very first run
            log_say("Robot will start moving in 3 seconds...", cfg.play_sounds, blocking=False)
            time.sleep(3)

        start_episode_time = time.perf_counter()
        frame_count = 0
        # Calculate max_frames based on episode_time_s and fps
        # If fps is None, use robot.fps. If robot.fps is also None, this will be an issue.
        # Assume robot.fps provides a fallback if cfg.fps is None.
        current_fps = cfg.fps or getattr(robot, 'fps', 30) # Default to 30 if robot has no fps attr
        if current_fps <= 0: current_fps = 30 # Ensure positive FPS

        max_frames_from_time = float('inf')
        if cfg.episode_time_s is not None and cfg.episode_time_s > 0:
            max_frames_from_time = int(cfg.episode_time_s * current_fps)

        while True:
            if events.get("stop_recording", False) or events.get("exit_early", False):
                if events.get("exit_early", False):
                    log_say("Exiting episode early.", cfg.play_sounds)
                    events["exit_early"] = False  # Reset for next episode
                break

            if frame_count >= max_frames_from_time:
                log_say("Episode time limit reached.", cfg.play_sounds)
                break

            loop_start_time = time.perf_counter()

            # GeminiPolicy does not use the observation in the batch for its prompt currently.
            # It might be used for device placement by base classes.
            # Passing an empty dict, assuming the policy handles it or doesn't need specific obs keys.
            action_input_batch = {}
            try:
                action = policy.select_action(action_input_batch)
            except Exception as e:
                logging.error(f"Error during policy.select_action: {e}")
                log_say("Error in policy, stopping episode.", cfg.play_sounds)
                break # Stop this episode on policy error
            
            # Ensure action is a numpy array on CPU for the robot
            if hasattr(action, 'cpu') and hasattr(action, 'numpy'):
                robot_action = action.cpu().numpy()
            elif isinstance(action, list):
                 robot_action = np.array(action, dtype=np.float32)  # Gemini sometimes returns list of lists
                 if robot_action.ndim == 2 and robot_action.shape[0] == 1:  # if it's [[...]]
                     robot_action = robot_action.squeeze(0)
            elif not isinstance(action, np.ndarray):
                logging.error(f"Policy action type {type(action)} not directly usable by robot.send_action.")
                log_say("Invalid action from policy, stopping episode.", cfg.play_sounds)
                break
            else:
                robot_action = action # Assuming it's already a numpy array

            # Ensure robot_action is a torch.Tensor (as expected by robot API)
            import torch
            if isinstance(robot_action, np.ndarray):
                robot_action = torch.from_numpy(robot_action.astype(np.float32))
            elif isinstance(robot_action, list):
                robot_action = torch.tensor(robot_action, dtype=torch.float32)

            try:
                robot.send_action(robot_action)
            except Exception as e:
                logging.error(f"Error during robot.send_action: {e}")
                log_say("Error sending action to robot, stopping episode.", cfg.play_sounds)
                break # Stop this episode on robot error

            if cfg.display_data and not is_headless():
                if hasattr(robot, 'publish_to_rerun'):
                    exec_dt = time.perf_counter() - loop_start_time
                    # Minimal observation for logging, if available
                    obs_for_rerun = robot.get_observation() if hasattr(robot, "get_observation") else None
                    robot.publish_to_rerun(action=robot_action, observation=obs_for_rerun, exec_dt=exec_dt)

            loop_dt_s = time.perf_counter() - loop_start_time
            target_dt_s = 1.0 / current_fps
            busy_wait(target_dt_s - loop_dt_s)

            actual_dt_s = time.perf_counter() - loop_start_time
            log_control_info(robot, actual_dt_s, fps=current_fps)
            frame_count += 1
        
        # Brief pause or cleanup before next episode or exit
        if robot.is_connected and hasattr(robot, 'teleop_safety_stop'):
             robot.teleop_safety_stop() # Stop any lingering motion
        time.sleep(0.5)

    log_say("Policy execution finished.", cfg.play_sounds, blocking=True)
    if listener is not None:
        listener.stop()
    # Final safety stop
    if robot.is_connected and hasattr(robot, 'teleop_safety_stop'):
        robot.teleop_safety_stop()


@safe_disconnect
def replay(
    robot: Robot,
    cfg: ReplayControlConfig,
):
    # TODO(rcadene, aliberts): refactor with control_loop, once `dataset` is an instance of LeRobotDataset
    # TODO(rcadene): Add option to record logs

    dataset = LeRobotDataset(cfg.repo_id, root=cfg.root, episodes=[cfg.episode])
    actions = dataset.hf_dataset.select_columns("action")

    if not robot.is_connected:
        robot.connect()

    log_say("Replaying episode", cfg.play_sounds, blocking=True)
    for idx in range(dataset.num_frames):
        start_episode_t = time.perf_counter()

        action = actions[idx]["action"]
        robot.send_action(action)

        dt_s = time.perf_counter() - start_episode_t
        busy_wait(1 / cfg.fps - dt_s)

        dt_s = time.perf_counter() - start_episode_t
        log_control_info(robot, dt_s, fps=cfg.fps)


def _init_rerun(control_config: ControlConfig, session_name: str = "lerobot_control_loop") -> None:
    """Initializes the Rerun SDK for visualizing the control loop.

    Args:
        control_config: Configuration determining data display and robot type.
        session_name: Rerun session name. Defaults to "lerobot_control_loop".

    Raises:
        ValueError: If viewer IP is missing for non-remote configurations with display enabled.
    """
    if (control_config.display_data and not is_headless()) or (
        control_config.display_data and isinstance(control_config, RemoteRobotConfig)
    ):
        # Configure Rerun flush batch size default to 8KB if not set
        batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
        os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size

        # Initialize Rerun based on configuration
        rr.init(session_name)
        if isinstance(control_config, RemoteRobotConfig):
            viewer_ip = control_config.viewer_ip
            viewer_port = control_config.viewer_port
            if not viewer_ip or not viewer_port:
                raise ValueError(
                    "Viewer IP & Port are required for remote config. Set via config file/CLI or disable control_config.display_data."
                )
            logging.info(f"Connecting to viewer at {viewer_ip}:{viewer_port}")
            rr.connect_tcp(f"{viewer_ip}:{viewer_port}")
        else:
            # Get memory limit for rerun viewer parameters
            memory_limit = os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "10%")
            rr.spawn(memory_limit=memory_limit)


@parser.wrap()
def control_robot(cfg: ControlPipelineConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    robot = make_robot_from_config(cfg.robot)

    # TODO(Steven): Blueprint for fixed window size

    if isinstance(cfg.control, CalibrateControlConfig):
        calibrate(robot, cfg.control)
    elif isinstance(cfg.control, TeleoperateControlConfig):
        _init_rerun(control_config=cfg.control, session_name="lerobot_control_loop_teleop")
        teleoperate(robot, cfg.control)
    elif isinstance(cfg.control, RecordControlConfig):
        _init_rerun(control_config=cfg.control, session_name="lerobot_control_loop_record")
        record(robot, cfg.control)
    elif isinstance(cfg.control, ExecutePolicyControlConfig):
        _init_rerun(control_config=cfg.control, session_name="lerobot_control_loop_execute_policy")
        execute_policy_on_robot(robot, cfg.control)
    elif isinstance(cfg.control, ReplayControlConfig):
        replay(robot, cfg.control)
    elif isinstance(cfg.control, RemoteRobotConfig):
        from lerobot.common.robot_devices.robots.lekiwi_remote import run_lekiwi

        _init_rerun(control_config=cfg.control, session_name="lerobot_control_loop_remote")
        run_lekiwi(cfg.robot)

    if robot.is_connected:
        # Disconnect manually to avoid a "Core dump" during process
        # termination due to camera threads not properly exiting.
        robot.disconnect()


if __name__ == "__main__":
    control_robot()
