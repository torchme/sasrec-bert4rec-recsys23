#! /bin/bash
#SBATCH --job-name="Subsample"
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-10:0
#SBATCH --mail-user=nseverin@hse.ru
#SBATCH --mail-type=ALL
#SBATCH --output="result"%j.out
#SBATCH --error="error"%j.out
#SBATCH --constraint="type_a|type_b|type_c|type_d"

# Executable
chmod +x src/help/get_subsample.py
python3  src/help/get_subsample.py