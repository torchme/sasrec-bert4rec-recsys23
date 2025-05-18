#! /bin/bash
#SBATCH --job-name="bert4rec-amazon_m2-baseline2-0-1.sh"
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-05:0
#SBATCH --mail-user=nseverin@hse.ru
#SBATCH --mail-type=ALL
#SBATCH --output=results/bert4rec/amazon_m2/baseline2/"result"%j.out
#SBATCH --error=results/bert4rec/amazon_m2/baseline2/"error"%j.out
#SBATCH --constraint="type_e"

# Executable
chmod +x experiments-2_0/bash/bert4rec/amazon_m2/baseline2/0-1.sh
srun experiments-2_0/bash/bert4rec/amazon_m2/baseline2/0-1.sh
