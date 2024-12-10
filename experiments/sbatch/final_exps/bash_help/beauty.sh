#!/bin/bash

# Specify the folder containing the files
FOLDER_PATH="experiments/bash/final_exps/beauty"

# Check if the folder exists
if [ ! -d "$FOLDER_PATH" ]; then
  echo "Folder does not exist: $FOLDER_PATH"
  exit 1
fi

# Iterate over all files in the folder
for file in "$FOLDER_PATH"/*; do
  # Ensure it's a file (not a directory or special file)
  if [ -f "$file" ]; then
    echo "Submitting $file to sbatch..."
    sbatch "$(realpath "$file")" # Pass the full path to sbatch
  else
    echo "Skipping non-file item: $file"
  fi
done

echo "All files processed."