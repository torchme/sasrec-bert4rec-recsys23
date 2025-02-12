#! /bin/bash
#SBATCH --job-name="sasrec-beauty-baseline-exps_with_loss_logging-2-2.sh"
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-10:0
#SBATCH --mail-user=nseverin@hse.ru
#SBATCH --mail-type=ALL
#SBATCH --output=results/sasrec/beauty/baseline/exps_with_loss_logging/"result"%j.out
#SBATCH --error=results/sasrec/beauty/baseline/exps_with_loss_logging/"error"%j.out
#SBATCH --constraint="type_a|type_b|type_c|type_e"

# Executable
chmod +x experiments-2_0/bash/sasrec/beauty/baseline/exps_with_loss_logging/2-2.sh
srun experiments-2_0/bash/sasrec/beauty/baseline/exps_with_loss_logging/2-2.sh
