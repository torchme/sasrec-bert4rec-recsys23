#!/bin/bash
. /home/nseverin/sasrec-bert4rec-recsys23/venv/bin/activate

PYTHONPATH=../../../../ python -m src.training --config "experiments/configs/kion_en/bert4rec.yaml"