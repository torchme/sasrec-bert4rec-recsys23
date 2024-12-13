#! /bin/bash
#SBATCH --job-name="Sasrec-REES-65-69.sh"
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1-10:0
#SBATCH --mail-user=nseverin@hse.ru
#SBATCH --mail-type=ALL
#SBATCH --output="result"%j.out
#SBATCH --error="error"%j.out
#SBATCH --constraint="type_a|type_b|type_c|type_e"

# Executable
chmod +x experiments/bash/final_exps/rees46/65-69.sh
srun experiments/bash/final_exps/rees46/65-69.sh