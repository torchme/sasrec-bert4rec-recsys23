#! /bin/bash
#SBATCH --job-name="sasrec-beauty-KION_INITIAL-other_seed-20-29.sh"
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-20:0
#SBATCH --mail-user=nseverin@hse.ru
#SBATCH --mail-type=ALL
#SBATCH --output=results/sasrec/beauty/KION_INITIAL/other_seed/"result"%j.out
#SBATCH --error=results/sasrec/beauty/KION_INITIAL/other_seed/"error"%j.out
#SBATCH --constraint="type_a|type_b|type_c|type_e"

# Executable
chmod +x experiments-2_0/bash/sasrec/beauty/KION_INITIAL/other_seed/20-29.sh
srun experiments-2_0/bash/sasrec/beauty/KION_INITIAL/other_seed/20-29.sh
