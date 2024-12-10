#!/bin/bash
. /home/nseverin/sasrec-bert4rec-recsys23/venv/bin/activate

config_path = "experiments/configs/final_sasrec_exps/rees46"
PYTHONPATH=../../../../ python -m src.process_config_batch --folder_path "$config_path" --start {start} --end {end}