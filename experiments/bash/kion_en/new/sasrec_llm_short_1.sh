#!/bin/bash
. /home/nseverin/sasrec-bert4rec-recsys23/venv/bin/activate

finetune_epochs=(10)
alpha=(0.4 0.6 0.8 0.9 0.95 1)

# Iterate over each combination of finetune_epochs and alpha
for epoch in "${finetune_epochs[@]}"; do
  for a in "${alpha[@]}"; do
    # Format alpha to remove the decimal point
    alpha_no_point=$(echo "$a" | sed 's/\.//g')

    # Construct the config file path
    config_file="experiments/configs/kion_en/short_1/sasrec_llm-${alpha_no_point}-${epoch}.yaml"

    # Run the command
    echo "Running: PYTHONPATH=../../../ python -m src.training --config $config_file"
    PYTHONPATH=../../../ python -m src.training --config "$config_file"
  done
done