#!/bin/bash
. /home/nseverin/sasrec-bert4rec-recsys23/venv/bin/activate

PYTHONPATH=../../../../ python -m src.process_config_batch --folder_path "experiments-2_0/configs/bert4rec/amazon_m2/M2_BERT_LLM_2/other_seed" --start 32 --end 39