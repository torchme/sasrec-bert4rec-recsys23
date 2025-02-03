#!/bin/bash
. /home/nseverin/sasrec-bert4rec-recsys23/venv/bin/activate

PYTHONPATH=../../../../ python -m src.process_config_batch --folder_path "experiments-2_0/configs/sasrec/kion_en/KION_NORMAL/other_seed" --start 24 --end 35