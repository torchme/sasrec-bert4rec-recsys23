import os
import argparse

import mlflow
import yaml

from src.training import process_config
from src.training_with_2stage_hist import train_model


def process_files_in_range(folder_path, start, end):
    # List all files in the directory
    end += 1
    files = os.listdir(folder_path)

    # Sort files alphabetically
    files.sort()

    # Extract the relevant range of files
    if start is None or start < 0:
        files_to_process = files[:end]
    elif end is None or end < 0:
        files_to_process = files[start:]
    else:
        files_to_process = files[start:end]

    # Process each file in the range
    for file in files_to_process:
        file_path = os.path.join(folder_path, file)

        if os.path.isfile(file_path):  # Check if it's a file
            print(f"Processing file: {file_path}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                best_model = train_model(config)
                config['fine_tune_epoch'] = 0
                config['data']['profile_train_sequences'] = config['data']['finetune_train_sequences']
                train_model(config, model=best_model)
            except Exception as e:
                print(e)
                print('Skipped config!')
                try:
                    # finishing run
                    mlflow.end_run()
                except Exception as e:
                    pass
        else:
            print(f"Skipping non-file: {file_path}")


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process files in a folder within a given index range.")
    parser.add_argument("--folder_path", type=str, help="Path to the folder containing the files.")
    parser.add_argument("--start", type=int, help="Starting index (inclusive).")
    parser.add_argument("--end", type=int, help="Ending index (exclusive).")

    # Parse arguments
    args = parser.parse_args()

    # Call the function with parsed arguments
    process_files_in_range(args.folder_path, args.start, args.end)


# process_files_in_range('experiments/configs/final_sasrec_exps/kion_en', None, 3)