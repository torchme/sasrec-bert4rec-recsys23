#! /bin/bash
#SBATCH --job-name="sasrec-kion_en-baseline_final_loss-other_seed-0-0.sh"
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-15:0
#SBATCH --mail-user=nseverin@hse.ru
#SBATCH --mail-type=ALL
#SBATCH --output=results/sasrec/kion_en/baseline_final_loss/other_seed/"result"%j.out
#SBATCH --error=results/sasrec/kion_en/baseline_final_loss/other_seed/"error"%j.out
#SBATCH --constraint="type_a|type_b|type_c|type_e"

# Executable
chmod +x experiments-2_0/bash/sasrec/kion_en/baseline_final_loss/other_seed/0-0.sh
srun experiments-2_0/bash/sasrec/kion_en/baseline_final_loss/other_seed/0-0.sh
