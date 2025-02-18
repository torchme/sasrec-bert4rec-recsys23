#! /bin/bash
#SBATCH --job-name="sasrec-beauty-base"
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-1:0
#SBATCH --mail-user=nseverin@hse.ru
#SBATCH --mail-type=ALL
#SBATCH --output=results/sasrec/beauty/baseline/single_seed/"result"%j.out
#SBATCH --error=results/sasrec/beauty/baseline/single_seed/"error"%j.out
#SBATCH --constraint="type_a|type_b|type_c"

# Executable
chmod +x experiments/bash/sasrec/last/beauty.sh
srun experiments/bash/sasrec/last/beauty.sh