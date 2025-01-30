#! /bin/bash
#SBATCH --job-name="sasrec-amazon_m2-baseline-single_seed-48-59.sh"
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-0:0
#SBATCH --mail-user=nseverin@hse.ru
#SBATCH --mail-type=ALL
#SBATCH --output=results/sasrec/amazon_m2/baseline/single_seed/"result"%j.out
#SBATCH --error=results/sasrec/amazon_m2/baseline/single_seed/"error"%j.out
#SBATCH --constraint="type_e"

# Executable
chmod +x experiments-2_0/bash/sasrec/amazon_m2/baseline/single_seed/48-59.sh
srun experiments-2_0/bash/sasrec/amazon_m2/baseline/single_seed/48-59.sh
