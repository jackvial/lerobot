#!/bin/bash

python lerobot/scripts/train.py   dataset_repo_id=jackvial/koch_pick_and_place_pistachio_8_e100   policy=act_koch_real   env=koch_real   hydra.run.dir=outputs/train/act_koch_pick_and_place_pistachio_8_e100_2   hydra.job.name=act_koch_pick_and_place_pistachio_8_e100_2   device=cuda   wandb.enable=false