#!/usr/bin/env bash

# Usage: ./merge_csv.sh output.csv input1.csv input2.csv ...

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 output.csv input1.csv [input2.csv ...]"
  exit 1
fi

OUTPUT="$1"
shift

FIRST_FILE="$1"

# Write header from the first file
head -n 1 "$FIRST_FILE" > "$OUTPUT"

# Append data rows from all files
for FILE in "$@"; do
  tail -n +2 "$FILE" >> "$OUTPUT"
done

echo "Merged CSV written to $OUTPUT"

