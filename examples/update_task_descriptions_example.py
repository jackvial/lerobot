#!/usr/bin/env python

"""
Example: Update Task Descriptions in a LeRobot Dataset

This example demonstrates how to update task descriptions/names in a LeRobot dataset.
"""

from lerobot.datasets.dataset_tools import update_task_descriptions
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def example_update_by_name():
    """Update task descriptions using task names."""
    # Load your dataset
    dataset = LeRobotDataset("my_dataset")
    
    # Print current tasks
    print("Current tasks:")
    for idx, task_name in enumerate(dataset.meta.tasks.index):
        print(f"  {idx}: {task_name}")
    
    # Update task descriptions by name
    task_mapping = {
        "Pick cube": "Pick up the red cube from the table",
        "Place cube": "Place the red cube in the blue bin",
    }
    
    print("\nUpdating tasks...")
    new_dataset = update_task_descriptions(
        dataset,
        task_mapping=task_mapping,
        output_dir="./my_dataset_updated",
        repo_id="my_dataset_updated"
    )
    
    # Print updated tasks
    print("\nUpdated tasks:")
    for idx, task_name in enumerate(new_dataset.meta.tasks.index):
        print(f"  {idx}: {task_name}")
    
    print(f"\nDataset saved to {new_dataset.root}")


def example_update_by_index():
    """Update task descriptions using task indices."""
    # Load your dataset
    dataset = LeRobotDataset("my_dataset")
    
    # Update task descriptions by index (more reliable if task names might change)
    task_mapping = {
        0: "First task with detailed description",
        1: "Second task with detailed description",
        2: "Third task with detailed description",
    }
    
    print("Updating tasks by index...")
    new_dataset = update_task_descriptions(
        dataset,
        task_mapping=task_mapping,
        output_dir="./my_dataset_updated",
        repo_id="my_dataset_updated"
    )
    
    print(f"Dataset saved to {new_dataset.root}")


def example_partial_update():
    """Update only some task descriptions."""
    # Load your dataset
    dataset = LeRobotDataset("my_dataset")
    
    # Only update task 0, leave others unchanged
    task_mapping = {
        "task_0": "Pick up the red cube (updated description)",
    }
    
    print("Updating only task_0...")
    new_dataset = update_task_descriptions(
        dataset,
        task_mapping=task_mapping,
        output_dir="./my_dataset_partial",
        repo_id="my_dataset_partial"
    )
    
    # Verify the changes
    print("\nTask updates:")
    print(f"  task_0 -> {new_dataset.meta.tasks.iloc[0].name}")
    print(f"  Other tasks remain unchanged")


def example_mixed_keys():
    """Update tasks using both names and indices."""
    # Load your dataset
    dataset = LeRobotDataset("my_dataset")
    
    # Mix string keys (task names) and int keys (task indices)
    task_mapping = {
        "task_0": "Updated by name",
        1: "Updated by index",
    }
    
    print("Updating tasks with mixed key types...")
    new_dataset = update_task_descriptions(
        dataset,
        task_mapping=task_mapping,
        output_dir="./my_dataset_mixed",
        repo_id="my_dataset_mixed"
    )
    
    print("Tasks updated successfully")


if __name__ == "__main__":
    # Uncomment the example you want to run
    
    # example_update_by_name()
    # example_update_by_index()
    # example_partial_update()
    # example_mixed_keys()
    
    print("Example script - uncomment the function you want to run")

