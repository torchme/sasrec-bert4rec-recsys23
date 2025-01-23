#! /bin/bash
#SBATCH --job-name="{model_name}-{dataset}-{experiment_name}-{seed_folder}-{bash_name}"
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time={time_to_take}
#SBATCH --mail-user=nseverin@hse.ru
#SBATCH --mail-type=ALL
#SBATCH --output={artefact_path}/"result"%j.out
#SBATCH --error={artefact_path}/"error"%j.out
#SBATCH --constraint="type_a|type_b|type_c|type_e"

# Executable
chmod +x experiments/bash/{model_name}/{dataset}/{experiment_name}/{seed_folder}/{bash_name}
srun experiments/bash/{model_name}/{dataset}/{experiment_name}/{seed_folder}/{bash_name}

# model_name dataset bash_name time_to_take artefact_path experiment_name seed_folder bash_name