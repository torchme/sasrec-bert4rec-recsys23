#!/bin/bash
. /home/nseverin/sasrec-bert4rec-recsys23/venv/bin/activate

PYTHONPATH=../../../../ python -m src.process_config_batch --folder_path "experiments-2_0/configs/sasrec/beauty/BEAUTY_INITIAL/single_seed" --start 30 --end 59