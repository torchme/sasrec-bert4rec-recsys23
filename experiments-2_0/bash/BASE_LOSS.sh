#!/bin/bash
. /home/nseverin/sasrec-bert4rec-recsys23/venv/bin/activate

PYTHONPATH=../../../../ python -m src.process_config_batch_with_loss --folder_path "{folder_path}" --start {start} --end {end}