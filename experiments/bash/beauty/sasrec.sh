#!/bin/bash
. /home/nseverin/sasrec-bert4rec-recsys23/venv/bin/activate

PYTHONPATH=../../../ python -m src.training --config experiments/configs/beauty/sasrec.yaml