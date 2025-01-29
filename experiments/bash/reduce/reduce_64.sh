#!/bin/bash
. /home/nseverin/sasrec-bert4rec-recsys23/venv/bin/activate

PYTHONPATH=../../../ python -m src.help.reduce_dim --config_path experiments/reduce_configs/reduce_dim64.json