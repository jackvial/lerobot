#!/usr/bin/env python

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
Merge multiple LeRobot datasets with the same prefix into a single dataset.

This script finds all datasets with a given prefix, downloads them, merges their episodes,
and uploads the combined dataset to the Hugging Face Hub.

Example usage:
```bash
python lerobot/scripts/merge_datasets.py --prefix "jackvial/koch_screwdriver_attach_orange_panel_"
```

This would find and merge all datasets like:
- jackvial/koch_screwdriver_attach_orange_panel_1
- jackvial/koch_screwdriver_attach_orange_panel_2
- etc.

The merged dataset will be uploaded as:
- jackvial/koch_screwdriver_attach_orange_panel_merged
"""

import argparse
import json
import logging
import shutil
import tempfile
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, list_datasets

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.push_dataset_to_hub.utils import check_repo_id
from lerobot.common.datasets.utils import (
    append_jsonlines, 
    write_episode_stats, 
    write_info
)
from lerobot.common.utils.utils import init_logging


def find_datasets_with_prefix(prefix: str) -> list[str]:
    """
    Find all datasets on Hugging Face Hub that start with the given prefix.
    
    Args:
        prefix: The prefix to search for (e.g., "username/dataset_prefix_")
    
    Returns:
        List of dataset repository IDs that match the prefix
    """
    logging.info(f"Searching for datasets with prefix: {prefix}")
    
    # Extract the user/org from the prefix
    if "/" not in prefix:
        raise ValueError("Prefix must include username/organization (e.g., 'username/prefix_')")
    
    user_org = prefix.split("/")[0]
    
    # Search for datasets
    datasets = list_datasets(author=user_org)
    
    matching_datasets = []
    for dataset in datasets:
        if dataset.id.startswith(prefix):
            matching_datasets.append(dataset.id)
    
    logging.info(f"Found {len(matching_datasets)} datasets matching prefix:")
    for dataset_id in sorted(matching_datasets):
        logging.info(f"  - {dataset_id}")
    
    return sorted(matching_datasets)


def merge_datasets(
    dataset_ids: list[str],
    output_repo_id: str,
    tolerance_s: float = 1e-4
) -> bool:
    """
    Merge multiple LeRobot datasets into a single dataset.
    
    Args:
        dataset_ids: List of dataset repository IDs to merge
        output_repo_id: Repository ID for the merged dataset
        tolerance_s: Tolerance for timestamp validation
    
    Returns:
        True if successful, False otherwise
    """
    if not dataset_ids:
        logging.error("No datasets to merge")
        return False
    
    try:
        logging.info(f"Starting merge of {len(dataset_ids)} datasets")
        
        # Load all datasets
        datasets = []
        for dataset_id in dataset_ids:
            logging.info(f"Loading dataset: {dataset_id}")
            dataset = LeRobotDataset(dataset_id, tolerance_s=tolerance_s)
            datasets.append(dataset)
        
        # Use the first dataset as the template for structure
        template_dataset = datasets[0]
        
        # Validate that all datasets have compatible structure
        for i, dataset in enumerate(datasets[1:], 1):
            if dataset.fps != template_dataset.fps:
                logging.error(f"FPS mismatch: {dataset_ids[0]} has {template_dataset.fps}, "
                            f"{dataset_ids[i]} has {dataset.fps}")
                return False
            
            if dataset.features.keys() != template_dataset.features.keys():
                logging.error(f"Feature mismatch between {dataset_ids[0]} and {dataset_ids[i]}")
                return False
            
            if dataset.meta.robot_type != template_dataset.meta.robot_type:
                logging.warning(f"Robot type differs: {dataset_ids[0]} has '{template_dataset.meta.robot_type}', "
                              f"{dataset_ids[i]} has '{dataset.meta.robot_type}'")
        
        logging.info("All datasets are compatible, proceeding with merge...")
        
        # Create the merged dataset
        merged_dataset = LeRobotDataset.create(
            repo_id=output_repo_id,
            fps=template_dataset.fps,
            robot_type=template_dataset.meta.robot_type,
            features=template_dataset.meta.info["features"],
            use_videos=len(template_dataset.meta.video_keys) > 0,
        )
        
        # Use a more efficient approach: directly manipulate the dataset structure
        # instead of frame-by-frame copying which causes feature mismatch issues
        logging.info("Merging datasets using efficient concatenation approach...")
        
        # Create temporary directory for merged dataset
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir) / output_repo_id.replace("/", "_")
            temp_root.mkdir(parents=True, exist_ok=True)
            (temp_root / "meta").mkdir(parents=True, exist_ok=True)
            
            # Update the merged dataset structure
            merged_dataset.root = temp_root
            merged_dataset.meta.root = temp_root
            
            # Collect all episodes and reindex them
            all_episodes = {}
            all_episode_stats = {}
            all_tasks = {}
            task_index_mapping = {}
            
            # Merge all HuggingFace datasets
            all_hf_datasets = []
            merged_episode_idx = 0
            
            for dataset_idx, dataset in enumerate(datasets):
                logging.info(f"Processing dataset {dataset_idx + 1}/{len(datasets)}: {dataset_ids[dataset_idx]}")
                
                # Add tasks from this dataset
                for task_idx, task in dataset.meta.tasks.items():
                    if task not in task_index_mapping:
                        new_task_idx = len(task_index_mapping)
                        task_index_mapping[task] = new_task_idx
                        all_tasks[new_task_idx] = task
                
                # Process episodes from this dataset
                for episode_idx in range(dataset.num_episodes):
                    # Get episode metadata
                    episode_data = dataset.meta.episodes[episode_idx].copy()
                    episode_data["episode_index"] = merged_episode_idx
                    all_episodes[merged_episode_idx] = episode_data
                    
                    # Get episode stats
                    if episode_idx in dataset.meta.episodes_stats:
                        all_episode_stats[merged_episode_idx] = dataset.meta.episodes_stats[episode_idx]
                    
                    # Get episode data range
                    from_idx = dataset.episode_data_index["from"][episode_idx]
                    to_idx = dataset.episode_data_index["to"][episode_idx]
                    
                    # Extract episode data and update indices
                    episode_hf_data = dataset.hf_dataset.select(range(from_idx, to_idx))
                    
                    # Update episode_index and task_index in the data
                    def update_indices(example):
                        example["episode_index"] = merged_episode_idx
                        # Update task_index based on task mapping
                        old_task_idx = example["task_index"]
                        # Convert tensor to int if needed
                        if hasattr(old_task_idx, 'item'):
                            old_task_idx = old_task_idx.item()
                        old_task = dataset.meta.tasks[old_task_idx]
                        example["task_index"] = task_index_mapping[old_task]
                        return example
                    
                    episode_hf_data = episode_hf_data.map(update_indices)
                    all_hf_datasets.append(episode_hf_data)
                    
                    merged_episode_idx += 1
            
            # Concatenate all HuggingFace datasets
            from datasets import concatenate_datasets
            merged_hf_dataset = concatenate_datasets(all_hf_datasets)
            
            # Update the merged dataset metadata
            merged_dataset.meta.episodes = all_episodes
            merged_dataset.meta.episodes_stats = all_episode_stats
            merged_dataset.meta.tasks = all_tasks
            merged_dataset.meta.task_to_task_index = {task: idx for idx, task in all_tasks.items()}
            
            # Update dataset info
            merged_dataset.meta.info["total_episodes"] = len(all_episodes)
            merged_dataset.meta.info["total_frames"] = len(merged_hf_dataset)
            merged_dataset.meta.info["total_tasks"] = len(all_tasks)
            merged_dataset.meta.info["splits"] = {"train": f"0:{len(all_episodes)}"}
            
            # Set the merged HuggingFace dataset
            merged_dataset.hf_dataset = merged_hf_dataset
            
            # Recalculate episode data index
            from lerobot.common.datasets.utils import get_episode_data_index
            merged_dataset.episode_data_index = get_episode_data_index(
                merged_dataset.meta.episodes, 
                list(range(len(all_episodes)))
            )
            
            # Fix data and video paths
            merged_dataset.meta.info["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
            if len(template_dataset.meta.video_keys) > 0:
                merged_dataset.meta.info["video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
            
            # Write all metadata files
            from lerobot.common.datasets.utils import write_info, append_jsonlines, write_episode_stats
            
            # Write info.json
            write_info(merged_dataset.meta.info, temp_root)
            
            # Write tasks.jsonl
            tasks_path = temp_root / "meta" / "tasks.jsonl"
            if tasks_path.exists():
                tasks_path.unlink()
            for task_index, task in merged_dataset.meta.tasks.items():
                task_dict = {"task_index": task_index, "task": task}
                append_jsonlines(task_dict, tasks_path)
            
            # Write episodes.jsonl
            episodes_path = temp_root / "meta" / "episodes.jsonl"
            if episodes_path.exists():
                episodes_path.unlink()
            for episode_index, episode_data in merged_dataset.meta.episodes.items():
                append_jsonlines(episode_data, episodes_path)
            
            # Write episodes_stats.jsonl
            for episode_index, episode_stats in merged_dataset.meta.episodes_stats.items():
                write_episode_stats(episode_index, episode_stats, temp_root)
            
            # Save the merged HuggingFace dataset files with chunk structure
            logging.info("Saving merged dataset files...")
            for episode_idx in range(len(all_episodes)):
                from_idx = merged_dataset.episode_data_index["from"][episode_idx]
                to_idx = merged_dataset.episode_data_index["to"][episode_idx]
                
                episode_data = merged_dataset.hf_dataset.select(range(from_idx, to_idx))
                
                # Save with chunk structure
                chunk_idx = episode_idx // merged_dataset.meta.info["chunks_size"]
                data_dir = temp_root / "data" / f"chunk-{chunk_idx:03d}"
                data_dir.mkdir(parents=True, exist_ok=True)
                
                episode_file = data_dir / f"episode_{episode_idx:06d}.parquet"
                episode_data.to_parquet(episode_file)
            
            # Copy video files if they exist with chunk structure
            if len(template_dataset.meta.video_keys) > 0:
                logging.info("Copying video files...")
                import shutil
                merged_episode_idx = 0
                
                for dataset_idx, dataset in enumerate(datasets):
                    for episode_idx in range(dataset.num_episodes):
                        chunk_idx = merged_episode_idx // merged_dataset.meta.info["chunks_size"]
                        
                        for video_key in dataset.meta.video_keys:
                            # Get original video path
                            old_video_path = dataset.meta.get_video_file_path(episode_idx, video_key)
                            old_full_path = dataset.root / old_video_path
                            
                            # New video path with chunk structure
                            new_video_dir = temp_root / "videos" / f"chunk-{chunk_idx:03d}" / video_key
                            new_video_dir.mkdir(parents=True, exist_ok=True)
                            new_full_path = new_video_dir / f"episode_{merged_episode_idx:06d}.mp4"
                            
                            if old_full_path.exists():
                                shutil.copy2(old_full_path, new_full_path)
                            else:
                                logging.warning(f"Video file not found: {old_full_path}")
                        
                        merged_episode_idx += 1
            
            # Create README.md with merge information
            logging.info("Creating README.md with merge information...")
            import json
            readme_content = f"""---
license: apache-2.0
task_categories:
- robotics
tags:
- LeRobot
configs:
- config_name: default
  data_files: data/*/*.parquet
---

# Merged LeRobot Dataset

This dataset was created by merging multiple LeRobot datasets using the [LeRobot](https://github.com/huggingface/lerobot) merge tool.

## Source Datasets

This merged dataset combines the following {len(dataset_ids)} datasets:

{chr(10).join(f"- [{dataset_id}](https://huggingface.co/datasets/{dataset_id})" for dataset_id in dataset_ids)}

## Dataset Statistics

- **Total Episodes**: {merged_dataset.meta.total_episodes}
- **Total Frames**: {merged_dataset.meta.total_frames}
- **Robot Type**: {merged_dataset.meta.robot_type}
- **FPS**: {merged_dataset.fps}

## Dataset Structure

[meta/info.json](meta/info.json):
```json
{json.dumps(merged_dataset.meta.info, indent=4)}
```

## Merge Details

- **Merge Date**: Generated automatically
- **Source Count**: {len(dataset_ids)} datasets
- **Episode Renumbering**: Episodes are renumbered sequentially starting from 0

## Citation

**BibTeX:**

```bibtex
[More Information Needed]
```
"""
            
            readme_path = temp_root / "README.md"
            with open(readme_path, "w") as f:
                f.write(readme_content)
            
            # Push to hub
            logging.info(f"Pushing merged dataset to hub: {output_repo_id}")
            merged_dataset.push_to_hub(
                license="apache-2.0",
                tags=["LeRobot", "robotics", "merged-dataset"],
            )
        
            logging.info(f"Successfully merged {len(dataset_ids)} datasets into {output_repo_id}")
            logging.info(f"Merged dataset contains {merged_dataset.meta.total_episodes} episodes "
                       f"and {merged_dataset.meta.total_frames} frames")
            logging.info(f"Dataset is available at: https://huggingface.co/datasets/{output_repo_id}")
        
        return True
        
    except Exception as e:
        logging.error(f"Failed to merge datasets: {str(e)}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple LeRobot datasets with the same prefix",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge all datasets with the given prefix
  python lerobot/scripts/merge_datasets.py --prefix "jackvial/koch_screwdriver_attach_orange_panel_"
  
  # Specify custom output name
  python lerobot/scripts/merge_datasets.py --prefix "user/dataset_prefix_" --output "user/merged_dataset"
  
  # Include tolerance for timestamp validation
  python lerobot/scripts/merge_datasets.py --prefix "user/prefix_" --tolerance-s 1e-3
        """
    )
    
    parser.add_argument(
        "--prefix",
        type=str,
        required=True,
        help="Prefix to search for datasets (e.g., 'username/dataset_prefix_')"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output repository ID for merged dataset. If not provided, will use prefix + 'merged'"
    )
    
    parser.add_argument(
        "--tolerance-s",
        type=float,
        default=1e-4,
        help="Tolerance in seconds for timestamp validation (default: 1e-4)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only find and list datasets, don't perform the merge"
    )
    
    args = parser.parse_args()
    
    # Initialize logging
    init_logging()
    
    # Validate prefix format
    if "/" not in args.prefix:
        logging.error("Prefix must include username/organization (e.g., 'username/prefix_')")
        return 1
    
    # Find datasets with the given prefix
    try:
        dataset_ids = find_datasets_with_prefix(args.prefix)
    except Exception as e:
        logging.error(f"Failed to search for datasets: {e}")
        return 1
    
    if not dataset_ids:
        logging.error(f"No datasets found with prefix: {args.prefix}")
        return 1
    
    if args.dry_run:
        logging.info("Dry run mode - would merge the following datasets:")
        for dataset_id in dataset_ids:
            logging.info(f"  - {dataset_id}")
        return 0
    
    # Determine output repository ID
    if args.output:
        output_repo_id = args.output
    else:
        # Remove trailing underscore and add 'merged'
        base_name = args.prefix.rstrip("_")
        output_repo_id = f"{base_name}_merged"
    
    # Validate output repo ID
    try:
        check_repo_id(output_repo_id)
    except ValueError as e:
        logging.error(f"Invalid output repository ID '{output_repo_id}': {e}")
        return 1
    
    logging.info(f"Will merge {len(dataset_ids)} datasets into: {output_repo_id}")
    
    # Perform the merge
    success = merge_datasets(dataset_ids, output_repo_id, args.tolerance_s)
    
    if success:
        logging.info("Merge completed successfully!")
        return 0
    else:
        logging.error("Merge failed!")
        return 1


if __name__ == "__main__":
    exit(main()) 