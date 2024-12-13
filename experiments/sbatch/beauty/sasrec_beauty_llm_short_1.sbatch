#! /bin/bash
#SBATCH --job-name="beauty-sasrec-llm-short-1"
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-23:0
#SBATCH --mail-user=nseverin@hse.ru
#SBATCH --mail-type=ALL
#SBATCH --output="result"%j.out
#SBATCH --error="error"%j.out
#SBATCH --constraint="type_a|type_b|type_c|type_e"

# Executable
chmod +x experiments/bash/beauty/new/sasrec_llm_short_1.sh
srun experiments/bash/beauty/new/sasrec_llm_short_1.sh
