import pandas as pd
import sys
import os
import math
import torch
import torch.nn.functional as F
import numpy as np
import logging

# Prevent Python from generating .pyc files (compiled bytecode files)
sys.dont_write_bytecode = True

# Import necessary modules and configuration settings
from prompts import *
from model import *
from utils import *

from transformers import BertTokenizer, BertModel

printHeader(f"Enrich Concepts with Synonyms")

hpoIDs = testIDs

# Load the dataset from a pickle file
data = readPickle(inputFile)

parents = [[]  for _ in range(0, len(hpoIDs))]
children = [[]  for _ in range(0, len(hpoIDs))]

with newProgress() as progress:

    task = newTask(progress, len(hpoIDs), "Get Parents and Children")

    for index in range(0, len(hpoIDs)):
        children[index] = getChildLabels(data, hpoIDs[index])
        parents[index] = getParentLabels(data, hpoIDs[index])
        progress.update(task, advance=1)

log("Removing unnecessary data...")
lostByFilter = len(data.index)
data = data[data[hpoidColumn].isin(hpoIDs)]
log(f"Removed {lostByFilter - len(data.index)} entries.")

data[systemColumn] = [""] * len(data.index)
data[countColumn] = [1] * len(data.index)

log("Removing empty data...")
data, lostByFilter = removeEmptyRows(data=data)
log(f"Removed {lostByFilter} entries.")

log(f"Left with {len(data.index)} entries.")

log("Adding Definitions.")

logging.getLogger("vllm").setLevel(logging.ERROR)

log(f"Set up the LLM ({model_id})...")
model = Model(model=model_id)

definitions = data.loc[(data[languageColumn] == "en") & 
            (data[classColumn] == definitionClass), hpoidColumn].tolist()

sourceDefinitionCount = len(data.loc[
    (data[languageColumn] == "en") & 
    (data[classColumn] == definitionClass), hpoidColumn].tolist())

if (sourceDefinitionCount > 0):
    log(f"Found {sourceDefinitionCount}/{len(hpoIDs)} English " \
        "definitions.")

    noDefinitions = hpoIDs.copy()

    for hpoID in definitions:
        noDefinitions.remove(hpoID)

    if (len(noDefinitions) > 0):

        instructions = [getPreTaskSystem()] * len(noDefinitions)
        log(f"{model.addPrompt(systemRole, instructions)} system instructions " \
            "added to the model.")

        messages = []

        for hpoID in noDefinitions:
            messages.append(getPreTaskPart1("".join(getElements(data, hpoID, "en",labelClass))))

        log(f"{model.addPrompt(userRole, messages)} prompts added to " \
            "the model. Start generating responses...")
        model.generate()

        messages = []
        for hpoID in noDefinitions:
            messages.append(getPreTaskPart2(parents[hpoIDs == hpoID]))

        c = model.addPrompt(userRole, messages)
        log(f"{c} prompts added to the model. Start generating responses...")
        model.generate()

        messages = []
        for hpoID in noDefinitions:
            messages.append(getPreTaskPart3(children[hpoIDs == hpoID]))

        c = model.addPrompt(userRole, messages)
        log(f"{c} prompts added to the model. Start generating responses...")
        model.generate()

        log(f"{model.addPrompt(userRole, [getPreTaskPart4()])} " \
            "prompts added to the model. Start generating responses...")
        model.generate()

        messageHistories = model.getMessageHistories()

        definitionTexts = []

        with newProgress() as progress:
            task = newTask(progress, len(messageHistories), "Processing gen. Text")

            for messageHistory in messageHistories:
                definitionTexts.append(messageHistory[-1][messageTextElement].
                    replace("\n", "").strip())
                progress.update(task, advance=1)

        formattedDefinition = pd.DataFrame({
            languageColumn  : ["en"] * len(definitionTexts),
            contentColumn   : definitionTexts,
            classColumn     : [enrichedSourceDefinitionClass] * len(definitionTexts),
            hpoidColumn     : noDefinitions
        })

        data = pd.concat([data, formattedDefinition])

        # Reset index after cleaning
        data = data.reset_index(drop=True)

        model.logPrompts()

log("Definition adding completed.")

model.reset()

#instructions = [getSystem()] * len(hpoIDs) * generateTimes
#log(f"{model.addPrompt(systemRole, instructions)} system instructions " \
#    "added to the model...")

# Add first instruction.
messages = []

with newProgress() as progress:

    task = newTask(progress, len(hpoIDs) * generateTimes, "Set System Prompt")

    for hpoID in hpoIDs:
        for _ in range(0, generateTimes):
            messages.append(getAlternativeEasyPrompt1(
                "".join(getElements(data, hpoID, [labelClass], "en")),
                "".join(getElements(data, hpoID, [definitionClass, enrichedSourceDefinitionClass], "en")),
                parents[hpoIDs.index(hpoID)],
                children[hpoIDs.index(hpoID)]
            ))
            progress.update(task, advance=1)

log(f"{model.addPrompt(userRole, messages)} prompts added to " \
    "the model. Start generating responses...")
model.generate()

log(f"{model.addPrompt(userRole, [getAlternativeEasyPrompt2()])} prompts " \
    "added to the model. Start generating responses...")
model.generate()

log(f"{model.addPrompt(userRole, [getAlternativeEasyPrompt3()])} prompts " \
    "added to the model. Start generating responses...")
model.generate()

log(f"{model.addPrompt(userRole, [getAlternativeEasyPrompt4()])} prompts " \
    "added to the model. Start generating responses...")
model.generate()

log(f"{model.addPrompt(userRole, [getAlternativeEasyPrompt4()])} prompts " \
    "added to the model. Start generating responses...")
model.generate()

model.logPrompts()
"""
messagesHistories = model.getMessageHistories()

synonymLists = [[]  for _ in range(0, len(messagesHistories))]
incorrectFormats = []

# Process each concept ID to enrich synonyms
with newProgress() as progress:
    task = newTask(progress, len(messagesHistories), "Processing Synonyms")
    
    for index, messagesHistory in enumerate(messagesHistories):
            
        answer = str(messagesHistory[-1][messageTextElement]).strip()
        answer = replaceQuotes(answer.replace("\n", "").replace(".", ""))

        tmp = list(set(formatting(answer)))
        synonymLists[index] = [str(t).lower() for t in tmp]
        if synonymLists[index] is not None and "" in synonymLists[index]:
            synonymLists[index].remove("")
        
        if len(synonymLists[index]) == 0:
            if len(answer) > 0:
                incorrectFormats.append(hpoIDs[int(index / generateTimes)])

        progress.update(task, advance=1)

log(f"Incorrect formats count: {len(incorrectFormats)}")
if (len(incorrectFormats) > 0):
    log(f"{incorrectFormats}")

result = [[] for _ in range(0, len(synonymLists))]

log("Consolidating results...")
for index, l in enumerate(synonymLists):
    result[index] = pd.DataFrame({
        hpoidColumn : [hpoIDs[math.floor(index / generateTimes)]] * len(l),
        contentColumn : l,
        classColumn : [enrichedSourceExactSynonymClass] * len(l),
        languageColumn : ["en"] * len(l),
        systemColumn : [model_id] * len(l),
        "round" : [(index % generateTimes) + 1] * len(l)
    })
log("Results consolidated.")

log("Removing duplicates...")
generated = pd.concat(result)
generated = generated.drop_duplicates(ignore_index=False).reset_index(drop=True)
log("Duplicates removed.")

log("Removing empty data...")
generated, lostByFilter = removeEmptyRows(data=generated)
log(f"Removed {lostByFilter} entries.")

gold = data[((data[classColumn] == labelClass) | (data[classColumn] == exactSynonymClass)) & (data[languageColumn] == "en")]
if "source" in gold.columns:
    gold = gold.drop(['source'], axis=1)
    
gold = gold.drop_duplicates(ignore_index=True).reset_index(drop=True)

for index in range(0, len(gold.index)):
    gold.loc[index, contentColumn] = str(gold.loc[index, contentColumn]).lower()

log("Load BERT Tokenizer...")
tokenizer = BertTokenizer.from_pretrained("google-bert/bert-large-uncased")
model = BertModel.from_pretrained("google-bert/bert-large-uncased")
log("BERT Tokenizer loaded.")

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

log("Generating embeddings...")
embeddingsGenerated = embed(generated[contentColumn].tolist())
generated.to_csv(outputFileGenerated, mode="a", header=not os.path.exists(outputFileGenerated), index=False)
pd.DataFrame(embeddingsGenerated.numpy()).to_csv(outputFileGeneratedEmbeddings, mode="a", header=not os.path.exists(outputFileGeneratedEmbeddings), index=False)

if not os.path.exists(outputFileGold):
    embeddingsGold = embed(gold[contentColumn].tolist())
    gold.to_csv(outputFileGold)
    pd.DataFrame(embeddingsGold.numpy()).to_csv(outputFileGoldEmbeddings, index = False)
log("Embeddings generated.")

log("Task completed.")
"""