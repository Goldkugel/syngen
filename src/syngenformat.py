import pandas   as pd
import numpy    as np
import sys
import os
import torch
import time

# Prevent Python from generating .pyc files (compiled bytecode files)
sys.dont_write_bytecode = True

# Import project-specific helpers, model wrappers, and configuration
from model      import *
from utils      import *

# HuggingFace BERT tokenizer and model for embedding generation
from transformers import BertTokenizer, BertModel

# Print a formatted header indicating the start of this processing stage
printHeader(f"Formatting and Embedding LLM Answers")
start_time = time.time()

# Only proceed if formatted input data exists
exitIfFileNotExist(inputFileGenerationFormatted)

# Load previously formatted LLM responses
data = readCSV(inputFileGenerationFormatted)

# One list of synonyms per row in the input data
synonymLists = [[] for _ in range(0, len(data.index))]

# Track HPO IDs whose LLM output could not be parsed correctly
incorrectFormats = []

# Process each LLM answer and extract synonyms
with newProgress() as progress:
    task = newTask(progress, len(data.index), "Formatting Answers")
    
    for index, messagesHistory in data.iterrows():

        label = "".join(getElements(data, messagesHistory[hpoidColumn],
                [labelClass]))
            
        synonymLists[index] = formatAnswerGeneration(
            str(messagesHistory[contentColumn]), label)

        if synonymLists[index] is None:
            incorrectFormats.append(messagesHistory[hpoidColumn])
            synonymLists[index] = []

        progress.update(task, advance=1)

    progress.refresh()

# Log statistics about malformed LLM outputs
log(f"Incorrect formats count: {len(incorrectFormats)}. Logging HPO Concepts.")
if len(incorrectFormats) > 0:
    for incorrect in list(set(incorrectFormats)):
        log(f"{incorrect}: {incorrectFormats.count(incorrect)}", cmdline = 
            False)

formattedResult = []

# Convert synonym lists into a normalized DataFrame
log("Consolidating results...")
for index, l in enumerate(synonymLists):
    if l is not None and len(l) > 0:

        formattedResult.append(pd.DataFrame({
            hpoidColumn   : [data.loc[index, hpoidColumn]]      * len(l),
            contentColumn : l,
            classColumn   : [enrichedSourceExactSynonymClass]   * len(l),
            systemColumn  : [data.loc[index, systemColumn]]     * len(l),
            roundColumn   : [data.loc[index, roundColumn]]      * len(l),
            typeColumn    : [""]                                * len(l)  
        }))

# Merge all per-row DataFrames into a single result
generated = pd.concat(formattedResult)
log("Results consolidated.")

# Remove empty or invalid content entries
log("Removing invalid row(s)...")
removedRows = len(generated.index)
generated[contentColumn] = generated[contentColumn].replace('', np.nan)
generated.dropna(subset = [contentColumn], inplace = True)
generated = generated.reset_index(drop = True)
log(f"{removedRows - len(generated.index)} invalid rows removed.")

# Remove duplicate synonym rows
log("Removing duplicates...")
generated = generated.drop_duplicates(ignore_index = False)
generated = generated.reset_index(drop = True)
log("Duplicates removed.")

# Remove rows with empty content after normalization
log("Removing empty data...")
generated, lostByFilter = removeEmptyRows(data = generated)
log(f"Removed {lostByFilter} entries.")

# Generate a BERT embedding for a single text input
def get_bert_embedding(text):
    inputs = tokenizer(
        text,
        return_tensors  = "pt",
        padding         = True,
        truncation      = True
    )
    with torch.no_grad():
        outputs = model(**inputs)
    # Use [CLS] token embedding as sentence representation
    return outputs.last_hidden_state[:, 0, :].squeeze()

# Generate embeddings for a list of strings with progress reporting
def embed(data: list) -> list:
    ret = [[] for _ in range(0, len(data))]

    with newProgress() as progress:
        task = newTask(progress, len(data), "Generate Embeddings")

        for index, l in enumerate(data):
            if l is not None and len(l) > 0:
                ret[index] = get_bert_embedding(l)
            else:
                ret[index] = None
            progress.update(task, advance = 1)

        progress.refresh()

    # Stack embeddings into a single tensor
    return torch.stack(ret)

if generateEmbeddings:
    # Load pretrained BERT model and tokenizer
    log(f"Load BERT Tokenizer ('{embedding_model_id}')...")
    tokenizer   = BertTokenizer.from_pretrained(embedding_model_id)
    model       = BertModel.from_pretrained(embedding_model_id)
    log(f"BERT Tokenizer loaded.")

    # Generate embeddings for gold-standard data if needed
    log("Generating embeddings...")
    if not os.path.exists(outputFileGenerationGoldEmbeddings) and \
            os.path.exists(outputFileGenerationGold):

        gold = readCSV(outputFileGenerationGold)

        embeddingsGold = embed(gold[contentColumn].tolist())

        writeCSV(pd.DataFrame(embeddingsGold.numpy()), 
            outputFileGenerationGoldEmbeddings)

    # Generate embeddings for newly generated synonyms
    embeddingsGenerated = embed(generated[contentColumn].tolist())

    writeCSV(pd.DataFrame(embeddingsGenerated.numpy()), 
        outputFileGenerationFormattedEmbeddings)

    log("Embeddings generated.")

# Persist generated data and embeddings to disk
writeCSV(generated, outputFileGenerationFormatted)

end_time = time.time()
elapsed_seconds = end_time - start_time
minutes = int(elapsed_seconds // 60)

# Final completion log
printHeader(f"Formatting and Embedding completed. [Minutes: {minutes}]")