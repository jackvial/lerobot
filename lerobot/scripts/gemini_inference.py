

import os
import re
import cv2
import matplotlib.pyplot as plt
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import json
import PIL
import google.generativeai as genai
import rerun as rr
# from faster_whisper import WhisperModel
# import sounddevice as sd
import numpy as np
import os
# from pydub import AudioSegment
import io

from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.common.policies.pi0fast.modeling_pi0fast import PI0FASTPolicy
from lerobot.common.robot_devices.cameras.configs import OpenCVCameraConfig
from lerobot.common.robot_devices.robots.utils import make_robot_from_config
from lerobot.common.robot_devices.control_utils import busy_wait, log_control_info
from lerobot.common.robot_devices.robots.configs import KochRobotConfig
from lerobot.common.utils.gemini_perception import tensor_to_pil, get_2D_bbox, parse_json, normalize_bbox_0to1, plot_bbox, get_target_bbox, get_random_targets, create_pick_place_lists



def setup_key_listener():
    events = {"exit_early": False, "rerecord_episode": False, 
              "stop_recording": False, "select_new_bbox": False}
    
    from pynput import keyboard
    
    def on_press(key):
        try:
            if key == keyboard.Key.right:
                print("Right arrow key pressed. Exiting loop...")
                events["exit_early"] = True
            elif key == keyboard.Key.space:
                print("Space key pressed. Selecting new bbox...")
                events["select_new_bbox"] = True
        except Exception as e:
            print(f"Error handling key press: {e}")
    
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    
    return listener, events

def return_bbox(boxes, idx):
    pick_t = boxes[idx]['box_2d']
    norm_pick_t = [p/1000 for p in pick_t]
    return norm_pick_t, pick_t

def select_bbox(boxes):
    selection = int(input("Enter the number of the bounding box you want to select: "))

    # Validate selection
    if 0 <= selection < len(boxes):return return_bbox(boxes,selection)
    else:print(f"Invalid selection. Please choose a number between 1 and {len(boxes)}")

def iterate_over_bbox(boxes):
    for i in range(len(boxes)):
        norm_pick_t, pick_t = return_bbox(boxes,i)
        yield norm_pick_t, pick_t




# Create camera config using proper config objects
cameras = {
    "laptop": OpenCVCameraConfig(
        camera_index=0,  # Built-in webcam
        fps=30,
        width=640,
        height=480
    ),
#    "top": OpenCVCameraConfig(
#        camera_index=1,  # iPhone camera
#        fps=30,
#        width=640,
#        height=480
#    )
}

robot_cfg = KochRobotConfig(
            cameras=cameras,
            mock=False, 
            robot_type="koch_robot"
        )
        
        # Create and connect robot
robot = make_robot_from_config(robot_cfg)
robot.connect()

def main():
    inference_time_s = 100
    fps = 30
    device = "cuda"  # TODO: On Mac, use "mps" or "cpu"
    BOX_AUGMENTATION = True
    
    user_task = "Pickup the orange lego block and place it in the blue bin"
    prompt = """User request: """ + user_task + """

    Analyze the provided image and understand what the user needs. Identify all relevant objects on the desk that would help fulfill this request.
    
    For example:
    - If the user wants to build something (like a red lego wall or wooden tower or lego plane), find all appropriate building pieces (Focus on identifying all red lego bricks on the desk that could be used for building a wall. Or in case of wooden tower, identify all wooden blocks on the desk that could be used for building a tower or all lego pieces on the desk that could be used to build a plane.)
    - If the user wants to clear the desk, identify all objects on the desk that need to be cleared
    - If the user wants specific colored items, focus on those colors
    - If the users mentions place location, identify the location of the object on the desk
    - If user is pointing to an object, identify the object that the user is pointing to.
    
    Ignore the robot arm itself if visible.
    Return your findings strictly as a JSON array of bounding boxes.
    Example format: [{"label": "red lego brick", "box_2d": [100, 200, 150, 280]}, {"label": "blue bin", "box_2d": [500, 600, 700, 850]}]"""
    
    observation = robot.capture_observation()
    pick, place, norm_pick, norm_place, pick_labels, place_labels = get_target_bbox(observation["observation.images.laptop"], prompt=prompt)
    print("Pick objects:")
    for i in range(len(pick)):print(pick_labels[i])
    print("Place location:")
    for i in range(len(place)):print(place_labels[i], place[i])
    pick_t, place_t, norm_pick_t, norm_place_t, pick, norm_pick, pick_target_label, place_target_label, pick_labels = get_random_targets(pick, place, norm_pick, norm_place, pick_labels, place_labels)
    single_task = f"Grasp {pick_target_label} and put it in {place_target_label}"
    listener, events = setup_key_listener()

    for _ in range(inference_time_s * fps):
        start_time = time.perf_counter()

        # Read the follower state and access the frames from the cameras
        observation = robot.capture_observation()
        observation["task"] = single_task
        observation["observation.state"] = torch.cat([observation["observation.state"], norm_pick_t, norm_place_t])
        # Convert to pytorch format: channel first and float32 in [0,1]
        # with batch dimension
        for name in observation:
            if name == "task":
                # Skip tensor operations for string values
                observation[name] = [observation[name]]  # Wrap in list to simulate batch dimension
                continue
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
        #print(action[-2])
        robot.send_action(action)
            
        dt_s = time.perf_counter() - start_time
        busy_wait(1 / fps - dt_s)
            
        if events["select_new_bbox"]:
            events["select_new_bbox"] = False
            pick_t, place_t, norm_pick_t, norm_place_t, pick, norm_pick, pick_target_label, place_target_label, pick_labels = get_random_targets(pick, place, norm_pick, norm_place, pick_labels, place_labels)
            single_task = f"Grasp {pick_target_label} and put it in {place_target_label}"
        
        


if __name__ == "__main__":
    main()