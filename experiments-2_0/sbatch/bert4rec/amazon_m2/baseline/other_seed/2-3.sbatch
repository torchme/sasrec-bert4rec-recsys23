#! /bin/bash
#SBATCH --job-name="bert4rec-amazon_m2-baseline-other_seed-2-3.sh"
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-05:0
#SBATCH --mail-user=nseverin@hse.ru
#SBATCH --mail-type=ALL
#SBATCH --output=results/bert4rec/amazon_m2/baseline/other_seed/"result"%j.out
#SBATCH --error=results/bert4rec/amazon_m2/baseline/other_seed/"error"%j.out
#SBATCH --constraint="type_a|type_b|type_c|type_e"

# Executable
chmod +x experiments-2_0/bash/bert4rec/amazon_m2/baseline/other_seed/2-3.sh
srun experiments-2_0/bash/bert4rec/amazon_m2/baseline/other_seed/2-3.sh
