#!/bin/bash
. /home/nseverin/sasrec-bert4rec-recsys23/venv/bin/activate

PYTHONPATH=../../../../ python -m src.process_config_batch_with_loss --folder_path "experiments-2_0/configs/sasrec/ml20m/baseline_final/exps_with_loss_logging" --start 2 --end 2