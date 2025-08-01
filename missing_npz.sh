#!/bin/bash
set -e 
# This script checks for .bag files that do not have corresponding .npz files
count=0
for f in *.bag; do
  if [[ ! -f "${f%.bag}.npz" ]]; then
    echo "$f"
    ((count++))
  fi
done
echo "Total missing .npz files: $count"
