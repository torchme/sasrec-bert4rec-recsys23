#!/bin/bash
. /home/nseverin/sasrec-bert4rec-recsys23/venv/bin/activate

PYTHONPATH=../../../../ python -m src.process_config_batch_with_loss --folder_path "experiments-2_0/configs/sasrec/amazon_m2/baseline_final_loss/other_seed" --start 0 --end 0