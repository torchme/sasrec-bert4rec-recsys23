#! /bin/bash
#SBATCH --job-name="sasrec-ml20m-baseline_cold_users-other_seed-9-11.sh"
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-10:0
#SBATCH --mail-user=nseverin@hse.ru
#SBATCH --mail-type=ALL
#SBATCH --output=results/sasrec/ml20m/baseline_cold_users/other_seed/"result"%j.out
#SBATCH --error=results/sasrec/ml20m/baseline_cold_users/other_seed/"error"%j.out
#SBATCH --constraint="type_a|type_b|type_c|type_e"

# Executable
chmod +x experiments-2_0/bash/sasrec/ml20m/baseline_cold_users/other_seed/9-11.sh
srun experiments-2_0/bash/sasrec/ml20m/baseline_cold_users/other_seed/9-11.sh
