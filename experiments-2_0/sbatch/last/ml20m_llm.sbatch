#! /bin/bash
#SBATCH --job-name="sasrec-ml20m-llm"
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-2:0
#SBATCH --mail-user=nseverin@hse.ru
#SBATCH --mail-type=ALL
#SBATCH --output=results/sasrec/beauty/baseline/single_seed/"result"%j.out
#SBATCH --error=results/sasrec/beauty/baseline/single_seed/"error"%j.out
#SBATCH --constraint="type_a|type_b|type_c"

# Executable
chmod +x experiments-2_0/bash/sasrec/last/ml20m_llm.sh
srun experiments-2_0/bash/sasrec/last/ml20m_llm.sh