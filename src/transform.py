from owlready2 import get_ontology

import pandas as pd
import sys
import time

# Prevent Python from generating .pyc files
sys.dont_write_bytecode = True

# Project-specific configuration and utility functions
from config import *
from utils  import *

# ------------------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------------------

printHeader("Transforming the Raw HPO Data")

# To track time.
start_time = time.time()

# ------------------------------------------------------------------------------
# Load Human Phenotype Ontology (HPO) from OWL file
# ------------------------------------------------------------------------------

# Ensure the input file exists before proceeding.
exitIfFileNotExist(inputFileTransformed)






# Load the ontology from the OWL file
printRead(inputFileTransformed)
hpo = get_ontology(inputFileTransformed).load()
printReadDone(inputFileTransformed)

# ------------------------------------------------------------------------------
# Extract ontology content into tabular structures
# ------------------------------------------------------------------------------

# This list will store multiple DataFrames, each representing a different
# type of extracted ontology information
data = []

# --- Synonyms and their types ---
log("Retrieving Synonyms...")
data.append(getSynonymsAndTypes(hpo))
log(f"{len(data[-1].index)} Synonyms retrieved.")

# --- Comments associated with HPO terms ---
log("Retrieving Comments...")
data.append(getComments(hpo))
log(f"{len(data[-1].index)} Comments retrieved.")

# --- Definitions of HPO terms ---
log("Retrieving Definitions...")
data.append(getDefinitions(hpo))
log(f"{len(data[-1].index)} Definitions retrieved.")

# --- Preferred labels for HPO terms ---
log("Retrieving Labels...")
data.append(getLabels(hpo))
log(f"{len(data[-1].index)} Labels retrieved.")

# --- Child (subclass) relationships ---
log("Retrieving Children...")
data.append(getChildren(hpo))
log(f"{len(data[-1].index)} Children retrieved.")

# --- External references (e.g., publications, databases) ---
log("Retrieving References...")
data.append(getReferences(hpo))
log(f"{len(data[-1].index)} References retrieved.")

# ------------------------------------------------------------------------------
# Merge all extracted information into a single DataFrame
# ------------------------------------------------------------------------------

log("Merging Data...")

# Combine all DataFrames, remove duplicates, and reset the index
data = (
    pd.concat(data)
      .drop_duplicates(inplace = False, ignore_index = True)
      .reset_index(drop = True)
)

log(f"Merging resulted in {len(data.index)} Lines of Data.")

# ------------------------------------------------------------------------------
# Clean data: remove rows with empty or missing content
# ------------------------------------------------------------------------------

rowCount = len(data.index)
log("Removing Empty Content Rows...")

# Remove rows where the content column is empty or NaN
data = (
    data[data[contentColumn] != ""]
    .dropna(subset = [contentColumn])
    .reset_index(drop = True)
)

log(f"Removed {rowCount - len(data.index)} Rows due to Empty Content.")

printRowCount(data)
printDataSummary(data)

log("Write full transformated Data...")
writeCSV(data, outputFileTransformedFull)
log("Full transformed Data written.")

# ------------------------------------------------------------------------------
# Filter data to keep only selected HPO concepts
# ------------------------------------------------------------------------------

if len(testIDs) > 0:
    log("Creating a reduced set of HPO Concepts...")

    rowCount = len(data.index)

    # If test IDs are provided, limit the data to:
    # - the test IDs themselves
    # - their children
    # - their parents
    if testIDs is not None and len(testIDs) > 0:
        parentIDs = data.loc[
            (data[contentColumn].isin(testIDs)) &
            (data[classColumn] == childrenClass),
            hpoidColumn
        ].tolist()

        childIDs = data.loc[
            (data[hpoidColumn].isin(testIDs)) &
            (data[classColumn] == childrenClass),
            contentColumn
        ].tolist()

        hpoIDs = list(set(testIDs + childIDs + parentIDs))

    # ------------------------------------------------------------------------------
    # Reduce content to only the selected HPO concepts
    # ------------------------------------------------------------------------------

    result = []

    # Use a progress bar to track processing of each HPO ID
    with newProgress() as progress:
        task = newTask(progress, len(hpoIDs), "Reduce Content")

        for hpoID in hpoIDs:
            # Collect all relevant entries for the given HPO ID
            result.append(checkEntries(data, hpoID))
            progress.update(task, advance = 1)

        progress.refresh()

    # Merge reduced content back into a single DataFrame
    log("Merging Data...")
    data = pd.concat(result).reset_index(drop = True)
    log(f"Content Reduced by {rowCount - len(data.index)} Rows.")

    # ------------------------------------------------------------------------------
    # Output summary statistics
    # ------------------------------------------------------------------------------

    printRowCount(data)
    printDataSummary(data)

    # ------------------------------------------------------------------------------
    # Persist transformed data to disk
    # ------------------------------------------------------------------------------

    log("Write reduced transformated Data...")
    writeCSV(data, outputFileTransformed)
    log("Reduced transformed Data written.")






minutes         = int((time.time() - start_time) // 60)

printHeader(f"Transforming completed [Minutes: {minutes}]")