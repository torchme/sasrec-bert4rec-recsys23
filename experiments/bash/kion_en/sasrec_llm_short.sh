#!/bin/bash
. /home/nseverin/sasrec-bert4rec-recsys23/venv/bin/activate

finetune_epochs=(15 25 35 50)
alpha=(0.4 0.8)

# Iterate over each combination of finetune_epochs and alpha
for epoch in "${finetune_epochs[@]}"; do
  for a in "${alpha[@]}"; do
    # Format alpha to remove the decimal point
    alpha_no_point=$(echo "$a" | sed 's/\.//g')

    # Construct the config file path
    config_file="experiments/configs/kion_en/sasrec_llm_short-${alpha_no_point}-${epoch}.yaml"

    # Run the command
    echo "Running: PYTHONPATH=../../../ python -m src.training --config $config_file"
    PYTHONPATH=../../../ python -m src.training --config "$config_file"
  done
done