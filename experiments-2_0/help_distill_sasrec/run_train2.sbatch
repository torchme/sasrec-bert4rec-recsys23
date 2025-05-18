#! /bin/bash
#SBATCH --job-name="sasrec-help_distill_sasrec-0-4.sh"
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-20:0
#SBATCH --mail-user=nseverin@hse.ru
#SBATCH --mail-type=ALL
#SBATCH --output=results/sasrec/run_train2/"result"%j.out
#SBATCH --error=results/sasrec/run_train2/"error"%j.out
#SBATCH --type_e"

# Executable
chmod +x experiments-2_0/help_distill_sasrec/run_train2.sh
srun experiments-2_0/help_distill_sasrec/run_train2.sh
