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
printHeader(f"Merging Generated Classifications")
start_time = time.time()







# Merging Generated Classes
if len(inputFileClassificationMerged) == 0:
    log(f"No '{csvFileFormat}' files with prefix " + 
        f"'{outputFileNameClassificationFormattedPrefix}' found in " + 
        f"'{outputFolderNameFormatted}'")
else:
    # Read and merge CSV files
    dataframes = []
    log("Reading files...")
    with newProgress() as progress:
        task = newTask(progress, len(inputFileClassificationMerged), 
            "Processing Files")

        for file in sorted(inputFileClassificationMerged):
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
    writeCSV(merged_df, outputFileClassificationMerged)

    log(f"Merged {len(inputFileClassificationMerged)} files into " + 
        f"'{os.path.basename(outputFileClassificationMerged)}'")







minutes         = int((time.time() - start_time) // 60)

# Print a formatted header indicating the end of this processing stage
printHeader(f"Data Merged [Minutes: {minutes}]")