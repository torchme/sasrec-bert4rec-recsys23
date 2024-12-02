#! /bin/bash
#SBATCH --job-name="kion-sasrec-llm"
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-23:0
#SBATCH --mail-user=nseverin@hse.ru
#SBATCH --mail-type=ALL
#SBATCH --output="result"%j.out
#SBATCH --error="error"%j.out
#SBATCH --constraint="type_a|type_b|type_c|type_e"

# Executable
chmod +x experiments/bash/rees46/sasrec_llm.sh
srun experiments/bash/rees46/sasrec_llm.sh