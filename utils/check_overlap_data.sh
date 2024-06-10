#!/bin/bash

# Directories to compare
DIR1="/Users/huytrq/Workspace/unicas/AIA&ML/Wrist-Fracture-Detection/results/test/images"
DIR2="/Users/huytrq/Downloads/images2"

# Find overlapping files
overlapping_files=$(comm -12 <(ls "$DIR1" | sort) <(ls "$DIR2" | sort))

# Check if there are overlapping files and print them
if [ -z "$overlapping_files" ]; then
  echo "No overlapping files found."
else
  echo "Overlapping files:"
  echo "$overlapping_files"
fi

