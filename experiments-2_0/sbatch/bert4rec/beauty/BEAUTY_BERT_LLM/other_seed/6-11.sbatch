#! /bin/bash
#SBATCH --job-name="bert4rec-beauty-BEAUTY_BERT_LLM-other_seed-6-11.sh"
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-0:0
#SBATCH --mail-user=nseverin@hse.ru
#SBATCH --mail-type=ALL
#SBATCH --output=results/bert4rec/beauty/BEAUTY_BERT_LLM/other_seed/"result"%j.out
#SBATCH --error=results/bert4rec/beauty/BEAUTY_BERT_LLM/other_seed/"error"%j.out
#SBATCH --constraint="type_a|type_b|type_c|type_e"

# Executable
chmod +x experiments-2_0/bash/bert4rec/beauty/BEAUTY_BERT_LLM/other_seed/6-11.sh
srun experiments-2_0/bash/bert4rec/beauty/BEAUTY_BERT_LLM/other_seed/6-11.sh
