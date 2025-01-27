#!/bin/bash
. /home/nseverin/sasrec-bert4rec-recsys23/venv/bin/activate

PYTHONPATH=../../../../ python -m src.process_config_batch_for_historcial_mix --folder_path "experiments-2_0/configs/sasrec/beauty/BEAUTY_50_50_single_new/other_seed" --start 0 --end 2