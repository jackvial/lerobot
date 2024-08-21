#!/bin/bash
JOB_NAME="act_koch_pick_and_place_pistachio_8_e100_2_and_10_e20_and_11_e20_002"

# Run the command with nohup
nohup python lerobot/scripts/train.py \
	'dataset_repo_id=[jackvial/koch_pick_and_place_pistachio_8_e10, jackvial/koch_pick_and_place_pistachio_10_e20, jackvial/koch_pick_and_place_pistachio_11_e20]' \
	policy=act_koch_real \
	env=koch_real \
	hydra.run.dir=outputs/train/$JOB_NAME \
	hydra.job.name=$JOB_NAME \
	device=cuda \
	wandb.enable=true \
	> ${JOB_NAME}_output.log 2>&1 &

# Print the process ID
echo "Process started with PID $!"
