#! /bin/bash
#SBATCH --job-name="Bert4rec-Beauty-orig"
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-23:0
#SBATCH --mail-user=nseverin@hse.ru
#SBATCH --mail-type=ALL
#SBATCH --output="result"%j.out
#SBATCH --error="error"%j.out
#SBATCH --constraint="type_a|type_b|type_c|type_e"

# Executable
chmod +x experiments/bash/final_exps/bert/beauty/orig.sh
srun experiments/bash/final_exps/bert/beauty/orig.sh