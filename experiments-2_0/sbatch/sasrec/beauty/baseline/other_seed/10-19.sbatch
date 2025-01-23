#! /bin/bash
#SBATCH --job-name="sasrec-beauty-baseline-other_seed-10-19.sh"
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1-15:0
#SBATCH --mail-user=nseverin@hse.ru
#SBATCH --mail-type=ALL
#SBATCH --output=results/sasrec/beauty/baseline/other_seed/"result"%j.out
#SBATCH --error=results/sasrec/beauty/baseline/other_seed/"error"%j.out
#SBATCH --constraint="type_a|type_b|type_c|type_e"

# Executable
chmod +x experiments/bash/sasrec/beauty/baseline/other_seed/10-19.sh
srun experiments/bash/sasrec/beauty/baseline/other_seed/10-19.sh
