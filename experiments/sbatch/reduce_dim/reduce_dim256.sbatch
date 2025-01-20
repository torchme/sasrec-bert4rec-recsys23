#! /bin/bash
#SBATCH --job-name="Reduce-256"
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-23:0
#SBATCH --mail-user=nseverin@hse.ru
#SBATCH --mail-type=ALL
#SBATCH --output="result"%j.out
#SBATCH --error="error"%j.out
#SBATCH --constraint="type_a|type_b|type_c|type_d"

# Executable
chmod +x experiments/bash/reduce/reduce_256.sh
srun experiments/bash/reduce/reduce_256.sh
