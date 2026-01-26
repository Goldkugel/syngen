import pandas as pd
import sys
import os
import torch
import json
import numpy as np

# Prevent Python from generating .pyc files (compiled bytecode files)
sys.dont_write_bytecode = True

# Import project-specific helpers, model wrappers, and configuration
from prompts import *
from utils import *

# Print a formatted header indicating the start of this processing stage
printHeader(f"Merging Data")

# Find matching CSV files in the directory
csv_files = [
    os.path.join(inputFolderFormatted, file)
    for file in os.listdir(inputFolderFormatted)
    if file.startswith(outputFileNameFormattedPrefix) and file.endswith(".csv")
]

if not csv_files:
    log(f"No CSV files with prefix '{outputFileNameFormattedPrefix}' found in {os.path.basename(inputFolderFormatted)}")
else:
    # Read and merge CSV files
    dataframes = []
    log("Reading files...")
    with newProgress() as progress:
        task = newTask(progress, len(csv_files), "Processing Files")

        for file in sorted(csv_files):
            dataframes.append(pd.read_csv(file))
            progress.update(task, advance=1)

    log("Files read.")
    log("Merging Data...")
    merged_df = pd.concat(dataframes, ignore_index=True)
    log("Data merged.")

    # Write merged CSV
    merged_df.to_csv(outputFileMergedTerms, index=False)

    log(f"Merged {len(csv_files)} files into '{os.path.basename(outputFileMergedTerms)}'")

# Find matching CSV files in the directory
csv_files = [
    os.path.join(inputFolderFormatted, file)
    for file in os.listdir(inputFolderFormatted)
    if file.startswith(outputFileNameFormattedEmbeddingsprefix) and file.endswith(".csv")
]

if not csv_files:
    log(f"No CSV files with prefix '{outputFileNameFormattedEmbeddingsprefix}' found in {os.path.basename(inputFolderFormatted)}")
else:
    # Read and merge CSV files
    dataframes = []
    log("Reading files...")
    with newProgress() as progress:
        task = newTask(progress, len(csv_files), "Processing Files")

        for file in sorted(csv_files):
            dataframes.append(pd.read_csv(file))
            progress.update(task, advance=1)

    # Read and merge CSV files
    log("Files read.")
    log("Merging Data...")
    merged_df = pd.concat(dataframes, ignore_index=True)
    log("Data merged.")
    
    # Write merged CSV
    merged_df.to_csv(outputFileMergedEmbeddings, index=False)
    log(f"Merged {len(csv_files)} files into '{os.path.basename(outputFileMergedEmbeddings)}'")

# Print a formatted header indicating the start of this processing stage
printHeader(f"Data Merged")
