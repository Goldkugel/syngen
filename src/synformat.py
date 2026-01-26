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
from model import *
from utils import *

# HuggingFace BERT tokenizer and model for embedding generation
from transformers import BertTokenizer, BertModel

# Print a formatted header indicating the start of this processing stage
printHeader(f"Formatting and Embedding LLM Answers")

# Only proceed if formatted input data exists
exitIfFileNotExist(inputFileFormatted)

# Load previously formatted LLM responses
data = readPickle(inputFileFormatted)

# One list of synonyms per row in the input data
synonymLists = [[] for _ in range(0, len(data.index))]

# Track HPO IDs whose LLM output could not be parsed correctly
incorrectFormats = []

# Process each LLM answer and extract synonyms
with newProgress() as progress:
    task = newTask(progress, len(data.index), "Processing Synonyms")
    
    for index, messagesHistory in data.iterrows():
            
        # Retrieve and normalize raw LLM answer text
        answer = str(messagesHistory[contentColumn]).strip()
        answer = answer.replace("```json", "")
        answer = answer.replace("```", "")
        answer = answer.replace("\n", "")
        answer = answer.replace("'", '"')

        # Attempt to isolate a JSON object in the response
        if "{" in answer and "}" in answer:
            answer = answer[answer.index("{"):answer.index("}") + 1]

            try:
                # Parse JSON content
                jsonAnswer = json.loads(answer)

                if jsonAnswer is not None:
                    # Expect a dictionary containing "exact_synonyms"
                    if isinstance(jsonAnswer, dict) and "exact_synonyms" in dict(jsonAnswer).keys():
                        synonymLists[index] = jsonAnswer["exact_synonyms"]

                        # Validate synonym list structure
                        if (
                            synonymLists[index] is not None
                            and isinstance(synonymLists[index], list)
                            and all(isinstance(item, str) for item in synonymLists[index])
                        ):
                            # Remove duplicates and empty strings
                            synonymLists[index] = list(set(synonymLists[index]))
                            if "" in synonymLists[index]:
                                synonymLists[index].remove("")

                            # Remove label if it appears among synonyms
                            label = "".join(
                                getElements(
                                    data,
                                    messagesHistory[hpoidColumn],
                                    [labelClass]
                                )
                            )
                            if label in synonymLists[index]:
                                synonymLists[index].remove(label)
                        else:
                            synonymLists[index] = []
                            incorrectFormats.append(messagesHistory[hpoidColumn])
                    else:
                        synonymLists[index] = []
                        incorrectFormats.append(messagesHistory[hpoidColumn])
                else:
                    synonymLists[index] = []
                    incorrectFormats.append(messagesHistory[hpoidColumn])

            except json.JSONDecodeError:
                # JSON parsing failed
                incorrectFormats.append(messagesHistory[hpoidColumn])
        else:
            # No JSON-like structure found
            incorrectFormats.append(messagesHistory[hpoidColumn])

        progress.update(task, advance=1)

# Log statistics about malformed LLM outputs
log(f"Incorrect formats count: {len(incorrectFormats)}")
if len(incorrectFormats) > 0:
    for incorrect in list(set(incorrectFormats)):
        log(f"{incorrect}: {incorrectFormats.count(incorrect)}")

formattedResult = []

# Convert synonym lists into a normalized DataFrame
log("Consolidating results...")
for index, l in enumerate(synonymLists):
    if l is not None and len(l) > 0:

        formattedResult.append(pd.DataFrame({
            hpoidColumn   : [data.loc[index, hpoidColumn]] * len(l),
            contentColumn : l,
            classColumn   : [enrichedSourceExactSynonymClass] * len(l),
            systemColumn  : [data.loc[index, systemColumn]] * len(l),
            roundColumn   : [data.loc[index, roundColumn]] * len(l),
            typeColumn    : [""] * len(l)  
        }))

# Merge all per-row DataFrames into a single result
generated = pd.concat(formattedResult)
log("Results consolidated.")

# Remove empty or invalid content entries
removedRows = len(generated.index)
log("Removing invalid row(s)...")
generated[contentColumn] = generated[contentColumn].replace('', np.nan)
generated.dropna(subset=[contentColumn], inplace=True)
generated = generated.reset_index(drop=True)
log(f"{removedRows - len(generated.index)} invalid rows removed.")

# Remove duplicate synonym rows
log("Removing duplicates...")
generated = generated.drop_duplicates(ignore_index=False).reset_index(drop=True)
log("Duplicates removed.")

# Remove rows with empty content after normalization
log("Removing empty data...")
generated, lostByFilter = removeEmptyRows(data=generated)
log(f"Removed {lostByFilter} entries.")

# Generate a BERT embedding for a single text input
def get_bert_embedding(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True
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
            progress.update(task, advance=1)

    # Stack embeddings into a single tensor
    return torch.stack(ret)

# Load pretrained BERT model and tokenizer
log("Load BERT Tokenizer...")
tokenizer = BertTokenizer.from_pretrained("google-bert/bert-large-uncased")
model = BertModel.from_pretrained("google-bert/bert-large-uncased")
log("BERT Tokenizer loaded.")

# Generate embeddings for gold-standard data if needed
log("Generating embeddings...")
if not os.path.exists(outputFileGoldEmbeddings) and os.path.exists(outputFileGold):
    gold = pd.read_csv(outputFileGold)
    embeddingsGold = embed(gold[contentColumn].tolist())
    pd.DataFrame(embeddingsGold.numpy()).to_csv(
        outputFileGoldEmbeddings,
        index=False
    )

# Generate embeddings for newly generated synonyms
embeddingsGenerated = embed(generated[contentColumn].tolist())

# Persist generated data and embeddings to disk
generated.to_csv(outputFileFormatted, mode="w", index=False)
pd.DataFrame(embeddingsGenerated.numpy()).to_csv(
    outputFileFormattedEmbeddings,
    mode="w",
    index=False
)
log("Embeddings generated.")

# Final completion log
printHeader("Formatting and Embedding completed.")