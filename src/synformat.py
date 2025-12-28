import pandas as pd
import sys
import os
import torch
import json

# Prevent Python from generating .pyc files (compiled bytecode files)
sys.dont_write_bytecode = True

# Import necessary modules and configuration settings
from prompts import *
from model import *
from utils import *

from transformers import BertTokenizer, BertModel

printHeader(f"Formatting LLM answers")

if os.path.exists(outputFileRawGenerated):
    log("Loading data...")
    data = pd.read_pickle(outputFileRawGenerated)
    log("Data loaded.")

    synonymLists = [[]  for _ in range(0, len(data.index))]
    incorrectFormats = []

    # Process each concept ID to enrich synonyms
    with newProgress() as progress:
        task = newTask(progress, len(data.index), "Processing Synonyms")
        
        for index, messagesHistory in data.iterrows():
                
            answer = str(messagesHistory[contentColumn]).strip()
            answer = answer.replace("```json", "")
            answer = answer.replace("```", "")
            answer = answer.replace("\n", "")
            answer = answer.replace("'", '"')
            if "{" in answer and "}" in answer:
                answer = answer[answer.index("{"):answer.index("}") + 1]

                try:
                    jsonAnswer = json.loads(answer)
                    if jsonAnswer is not None:
                        if isinstance(jsonAnswer, dict) and "exact_synonyms" in dict(jsonAnswer).keys():
                            synonymLists[index] = jsonAnswer["exact_synonyms"]
                            if synonymLists[index] is not None and isinstance(synonymLists[index], list):
                                synonymLists[index] = list(set(synonymLists[index]))
                                if "" in synonymLists[index]:
                                    synonymLists[index].remove("")
                                label = "".join(getElements(data, messagesHistory[hpoidColumn], [labelClass], "en"))
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
                except json.JSONDecodeError as e:
                    incorrectFormats.append(messagesHistory[hpoidColumn])
            else:
                incorrectFormats.append(messagesHistory[hpoidColumn])
            progress.update(task, advance=1)

    log(f"Incorrect formats count: {len(incorrectFormats)}")
    if (len(incorrectFormats) > 0):
        for incorrect in list(set(incorrectFormats)):
            log(f"{incorrect}: {incorrectFormats.count(incorrect)}")

    formattedResult = [[] for _ in range(0, len(synonymLists))]

    log("Consolidating results...")
    for index, l in enumerate(synonymLists):
        if l is not None and len(l) > 0:
            formattedResult[index] = pd.DataFrame({
                hpoidColumn     : [data.loc[index, hpoidColumn]] * len(l),
                contentColumn   : l,
                classColumn     : [enrichedSourceExactSynonymClass] * len(l),
                languageColumn  : ["en"] * len(l),
                systemColumn    : [data.loc[index, systemColumn]] * len(l),
                roundColumn     : [data.loc[index, roundColumn]] * len(l)
            })
    generated = pd.concat(formattedResult)
    log("Results consolidated.")

    log("Removing duplicates...")
    generated = generated.drop_duplicates(ignore_index=False).reset_index(drop=True)
    log("Duplicates removed.")

    log("Removing empty data...")
    generated, lostByFilter = removeEmptyRows(data=generated)
    log(f"Removed {lostByFilter} entries.")

    def get_bert_embedding(text):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze()

    def embed(data : list) -> list:
        ret = [[] for _ in range(0, len(data))]

        with newProgress() as progress:

            task = newTask(progress, len(data), "Generate Embeddings")

            for index, l in enumerate(data):
                ret[index] = get_bert_embedding(l)
                progress.update(task, advance=1)

        return torch.stack(ret)

    log("Load BERT Tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("google-bert/bert-large-uncased")
    model = BertModel.from_pretrained("google-bert/bert-large-uncased")
    log("BERT Tokenizer loaded.")

    log("Generating embeddings...")
    if not os.path.exists(outputFileGoldEmbeddings) and os.path.exists(outputFileGold):
        gold = pd.read_csv(outputFileGold)
        embeddingsGold = embed(gold[contentColumn].tolist())
        pd.DataFrame(embeddingsGold.numpy()).to_csv(outputFileGoldEmbeddings, index = False)

    embeddingsGenerated = embed(generated[contentColumn].tolist())
    generated.to_csv(outputFileGenerated, mode="w", index=False)
    pd.DataFrame(embeddingsGenerated.numpy()).to_csv(outputFileGeneratedEmbeddings, mode="w", index=False)
    log("Embeddings generated.")
else:
    log("No raw data to format found.")

log("Task completed.")