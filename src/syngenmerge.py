import pandas as pd
import sys
import os
import time

# Prevent Python from generating .pyc files (compiled bytecode files)
sys.dont_write_bytecode = True

# Import project-specific helpers, model wrappers, and configuration
from prompts    import *
from utils      import *

# Print a formatted header indicating the start of this processing stage
printHeader(f"Merging Generated Synonyms")
start_time = time.time()

# Merging Generated Synonyms
if len(inputFileGenerationMerged) == 0:
    log(f"No '{csvFileFormat}' files with prefix " + 
            f"'{outputFileNameGenerationFormattedPrefix}' found in " +
            f"'{outputFolderNameFormatted}'")
else:
    # Read and merge CSV files
    dataframes = []
    log("Reading files...")
    with newProgress() as progress:
        task = newTask(progress, len(inputFileGenerationMerged), 
            "Processing Files")

        for file in sorted(inputFileGenerationMerged):
            dataframes.append(pd.read_csv(file))
            progress.update(task, advance = 1)

        progress.refresh()

    # Read and merge CSV files
    log("Files read.")

    log("Merging Data...")
    merged_df = pd.concat(dataframes, ignore_index = True)
    merged_df = merged_df.reset_index(drop = True)
    log("Data merged.")

    # Write merged CSV
    writeCSV(merged_df, outputFileGenerationMerged)

    log(f"Merged {len(inputFileGenerationMerged)} files into " + 
        f"'{os.path.basename(outputFileGenerationMerged)}'")

# Merging the Embeddings of the Generated Synonyms
if len(inputFileGenerationMergedEmbeddings) == 0:
    log(f"No CSV files with prefix " + 
        f"'{outputFileNameGenerationFormattedEmbeddingsPrefix}' found in " + 
        f"'{outputFolderNameFormatted}'")
else:
    # Read and merge CSV files
    dataframes = []
    log("Reading files...")
    with newProgress() as progress:
        task = newTask(progress, len(inputFileGenerationMergedEmbeddings), 
            "Processing Files")

        for file in sorted(inputFileGenerationMergedEmbeddings):
            dataframes.append(pd.read_csv(file))
            progress.update(task, advance=1)

    # Read and merge CSV files
    log("Files read.")

    log("Merging Data...")
    merged_df = pd.concat(dataframes, ignore_index = True)
    merged_df = merged_df.reset_index(drop = True)
    log("Data merged.")
    
    # Write merged CSV
    writeCSV(merged_df, outputFileGenerationMergedEmbeddings)
    log(f"Merged {len(inputFileGenerationMergedEmbeddings)} files into " + 
        f"'{os.path.basename(outputFileGenerationMergedEmbeddings)}'")

end_time = time.time()
elapsed_seconds = end_time - start_time
minutes = int(elapsed_seconds // 60)

# Print a formatted header indicating the end of this processing stage
printHeader(f"Data Merged [Minutes: {minutes}]")