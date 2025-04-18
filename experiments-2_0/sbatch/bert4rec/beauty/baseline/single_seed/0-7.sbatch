#! /bin/bash
#SBATCH --job-name="bert4rec-beauty-baseline-single_seed-0-7.sh"
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-16:0
#SBATCH --mail-user=nseverin@hse.ru
#SBATCH --mail-type=ALL
#SBATCH --output=results/bert4rec/beauty/baseline/single_seed/"result"%j.out
#SBATCH --error=results/bert4rec/beauty/baseline/single_seed/"error"%j.out
#SBATCH --constraint="type_e"

# Executable
chmod +x experiments-2_0/bash/bert4rec/beauty/baseline/single_seed/0-7.sh
srun experiments-2_0/bash/bert4rec/beauty/baseline/single_seed/0-7.sh
