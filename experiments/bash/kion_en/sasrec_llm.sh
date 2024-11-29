#!/bin/bash
. /home/nseverin/miniconda3/envs/recsys_seq_env/bin/activate

PYTHONPATH=../../../ python -m src.training --config experiments/configs/kion_en/sasrec_llm.yaml