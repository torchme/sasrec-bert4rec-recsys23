#! /bin/bash
#SBATCH --job-name="Sasrec-KION-32-39.sh"
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1-15:0
#SBATCH --mail-user=nseverin@hse.ru
#SBATCH --mail-type=ALL
#SBATCH --output="result"%j.out
#SBATCH --error="error"%j.out
#SBATCH --constraint="type_a|type_b|type_c|type_e"

# Executable
chmod +x experiments/bash/new_exps_2025/kion_en/32-39.sh
srun experiments/bash/new_exps_2025/kion_en/32-39.sh