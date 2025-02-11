#! /bin/bash
#SBATCH --job-name="sasrec-ml20m-ML20M_LLM_ALPHA_EXPS-other_seed-3-3.sh"
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-12:0
#SBATCH --mail-user=nseverin@hse.ru
#SBATCH --mail-type=ALL
#SBATCH --output=results/sasrec/ml20m/ML20M_LLM_ALPHA_EXPS/other_seed/"result"%j.out
#SBATCH --error=results/sasrec/ml20m/ML20M_LLM_ALPHA_EXPS/other_seed/"error"%j.out
#SBATCH --constraint="type_a|type_b|type_c|type_e"

# Executable
chmod +x experiments-2_0/bash/sasrec/ml20m/ML20M_LLM_ALPHA_EXPS/other_seed/3-3.sh
srun experiments-2_0/bash/sasrec/ml20m/ML20M_LLM_ALPHA_EXPS/other_seed/3-3.sh
