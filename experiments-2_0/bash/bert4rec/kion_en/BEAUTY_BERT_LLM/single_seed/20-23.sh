#!/bin/bash
. /home/nseverin/sasrec-bert4rec-recsys23/venv/bin/activate

PYTHONPATH=../../../../ python -m src.process_config_batch --folder_path "experiments-2_0/configs/bert4rec/kion_en/BEAUTY_BERT_LLM/single_seed" --start 20 --end 23