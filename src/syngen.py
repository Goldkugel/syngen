import pandas as pd
import sys
import os
import logging

# Prevent Python from generating .pyc files (compiled bytecode files)
sys.dont_write_bytecode = True

# Import necessary modules and configuration settings
from prompts import *
from model import *
from utils import *

printHeader(f"Enrich Concepts with Synonyms")

# Load the dataset from a pickle file
log("Loading data...")
data = readPickle(inputFile)
log("Data loaded.")

parents = {}
children = {}

with newProgress() as progress:

    task = newTask(progress, len(testIDs), "Get Parents and Children")

    for hpoID in testIDs:
        children[hpoID] = getChildLabels(data, hpoID)
        parents[hpoID] = getParentLabels(data, hpoID)
        progress.update(task, advance=1)

log("Removing unnecessary data...")
lostByFilter = len(data.index)
data = data[data[hpoidColumn].isin(testIDs)]
log(f"Removed {lostByFilter - len(data.index)} entries.")

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

log(f"Found {len(definitions)}/{len(testIDs)} English definitions.")

if (len(definitions) < len(testIDs)):
    noDefinitions = testIDs.copy()

    for hpoID in definitions:
        noDefinitions.remove(hpoID)

    log(f"Generating {len(noDefinitions)} missing definitions...")

    messages = []

    for hpoID in noDefinitions:
        messages.append(getPreTaskPart1("".join(getElements(data, hpoID, labelClass))))

    log(f"{model.addPrompt(userRole, messages)} prompts added to " \
        "the model. Start generating responses...")
    model.generate()

    messages = []
    for hpoID in noDefinitions:
        messages.append(getPreTaskPart2(parents[hpoID]))

    c = model.addPrompt(userRole, messages)
    log(f"{c} prompts added to the model. Start generating responses...")
    model.generate()

    messages = []
    for hpoID in noDefinitions:
        messages.append(getPreTaskPart3(children[hpoID]))

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
    model.reset()
else:
    log("No definition generation required.")

log("Definition adding completed.")

messages = []

with newProgress() as progress:

    task = newTask(progress, len(testIDs) * generateTimes, "Set System Prompt")

    for hpoID in testIDs:
        for _ in range(0, generateTimes):
            messages.append(getAlternativeComplexPrompt1(
                "".join(getElements(data, hpoID, [labelClass], "en")),
                "".join(getElements(data, hpoID, [definitionClass, enrichedSourceDefinitionClass], "en")),
                parents[hpoID],
                children[hpoID]
            ))
            progress.update(task, advance=1)

addedPrompts = model.addPrompt(userRole, messages)
log(f"{addedPrompts} prompts added. Start generating responses...")
model.generate()

addedPrompts = model.addPrompt(userRole, [getAlternativeComplexPrompt2()])
log(f"{addedPrompts} prompts added. Start generating responses...")
model.generate()

addedPrompts = model.addPrompt(userRole, [getAlternativeComplexPrompt3()])
log(f"{addedPrompts} prompts added. Start generating responses...")
model.generate()

addedPrompts = model.addPrompt(userRole, [getAlternativeComplexPrompt4()])
log(f"{addedPrompts} prompts added. Start generating responses...")
model.generate()

addedPrompts = model.addPrompt(userRole, [getAlternativeComplexPrompt5()])
log(f"{addedPrompts} prompts added. Start generating responses...")
model.generate()

model.logPrompts()

messagesHistories       = model.getMessageHistories()
generatedText           = [""]  * len(messagesHistories)
generatedTextHPOid      = [""]  * len(messagesHistories)
generatedRound          = [0]   * len(messagesHistories)

# Process each concept ID to enrich synonyms
with newProgress() as progress:
    task = newTask(progress, len(messagesHistories), "Processing Synonyms")
    
    for hpoID in testIDs:

        index = testIDs.index(hpoID)
        for t in range(0, generateTimes):
            generatedTextHPOid[index * generateTimes + t]   = hpoID
            messagesHistory                                 = messagesHistories[index * generateTimes + t]
            generatedText[index * generateTimes + t]        = str(messagesHistory[-1][messageTextElement])
            generatedRound[index * generateTimes + t]       = t + 1

            progress.update(task, advance=1)

log("Saving generated text...")
result = pd.DataFrame({
    contentColumn   : generatedText, 
    hpoidColumn     : generatedTextHPOid, 
    roundColumn     : generatedRound,
    classColumn     : [enrichedSourceExactSynonymClass] * len(generatedRound),
    languageColumn  : ["en"]                            * len(generatedRound),
    systemColumn    : [model_id]                        * len(generatedRound),
})
result.to_pickle(outputFileRawGenerated)
log("Saved generated text.")

if not os.path.exists(outputFileGold):
    log("Creating gold standard file...")
    gold = data[((data[classColumn] == labelClass) | (data[classColumn] == exactSynonymClass)) & (data[languageColumn] == "en")]
    if "source" in gold.columns:
        gold = gold.drop(['source'], axis=1)
    gold = gold.drop_duplicates(ignore_index=True).reset_index(drop=True)
    gold.to_csv(outputFileGold)
    log("Created gold standard file.")
else:
    log("Gold standard file found, nothing to create.")

log("Task completed.")