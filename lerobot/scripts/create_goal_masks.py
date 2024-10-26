"""
Draw goal masks and in-bounds mask.

How to use:
1. Put the cube on the bottom-left-most corner of the left goal region.
2. Run `python lerobot/scripts/create_goal_masks.py`
3. Draw the left goal mask on the opencv window that pops up (click-drag).
4. Move the cube to the bottom-right-most corner of the right goal region.
5. With the opencv window focussed, press any key.
6. Draw the right goal mask.
7. With the opencv window focussed, press any key.
8. Draw the mask for the whole in-bounds region.
9. With the opencv window focussed, press any key.
"""

from hydra.utils import instantiate
from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera
from lerobot.common.utils.utils import init_hydra_config
from lerobot.common.vision import GoalSetter
from pathlib import Path
import argparse
import numpy as np


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Create goal masks for robot vision")
    parser.add_argument(
        "position",
        type=str,
        choices=["left", "right", "center"],
        help="Position to create mask for (left, right, or center)",
    )
    args = parser.parse_args()

    # Initialize config
    cfg = init_hydra_config("lerobot/configs/robot/koch_tdmpc_jack.yaml")
    assert len(cfg["cameras"]) == 1

    # Initialize camera
    camera: OpenCVCamera = instantiate(cfg["cameras"][list(cfg["cameras"])[0]])
    camera.connect()

    print(f"Draw goal region for {args.position}")
    save_goal_mask_path = Path(f"/home/jack/code/rl/lerobot/outputs/goal_mask_{args.position}.npy")

    # Create and save mask
    goal_setter = GoalSetter()
    img = camera.async_read()
    goal_setter.set_image(img, resize_factor=8)
    print("Press q when done creating the mask")
    goal_setter.run()
    goal_mask = goal_setter.get_goal_mask()
    # goal_setter.save_goal_mask(save_goal_mask_path)
    print(f"Saving mask to {save_goal_mask_path}")
    print(f"Mask shape: {goal_mask.shape}")
    print(f"Mask dtype: {goal_mask.dtype}")

    sum_mask = goal_mask.sum()
    print(f"Sum of mask: {sum_mask}")

    # numpy version
    print("np.__version__", np.__version__)

    # Get the boolean mask and ensure it's a numpy array with correct dtype
    mask = goal_setter.get_goal_mask().astype(np.bool_)
    
    # Try saving with a different approach
    with open(save_goal_mask_path, 'wb') as f:
        np.save(f, mask, allow_pickle=False)
    
    print(f"Successfully saved mask to {save_goal_mask_path}")


if __name__ == "__main__":
    main()
