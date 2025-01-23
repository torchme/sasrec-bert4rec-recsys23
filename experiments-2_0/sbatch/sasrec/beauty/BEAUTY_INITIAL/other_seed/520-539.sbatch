#! /bin/bash
#SBATCH --job-name="sasrec-beauty-BEAUTY_INITIAL-other_seed-520-539.sh"
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-10:0
#SBATCH --mail-user=nseverin@hse.ru
#SBATCH --mail-type=ALL
#SBATCH --output=results/sasrec/beauty/BEAUTY_INITIAL/other_seed/"result"%j.out
#SBATCH --error=results/sasrec/beauty/BEAUTY_INITIAL/other_seed/"error"%j.out
#SBATCH --constraint="type_a|type_b|type_c|type_e"

# Executable
chmod +x experiments-2_0/bash/sasrec/beauty/BEAUTY_INITIAL/other_seed/520-539.sh
srun experiments-2_0/bash/sasrec/beauty/BEAUTY_INITIAL/other_seed/520-539.sh
