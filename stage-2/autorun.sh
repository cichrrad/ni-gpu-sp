#!/bin/bash

# Specify the directory where files are located
directory="../image-dataset"

# Specify the command to run on each file
# For example, to list details about each file:
command_to_run="build/canny"

# Check if the directory exists
if [ ! -d "$directory" ]; then
  echo "Error: Directory '$directory' does not exist."
  exit 1
fi

mkdir OUT
# Iterate over each file in the directory
for file in "$directory"/*; do
  # Only process if it is a regular file
  if [ -f "$file" ]; then
    echo "Processing file: $file"
    # extract just filename
    filename=$(basename "$file")
    # Execute the specified command on the file
    ${command_to_run} "$file" "OUT/out_${filename}" custom
    ${command_to_run} "$file" "OUT/out_openCV_${filename}" opencv
  fi
done
