import pandas as pd
import sys
import os
import logging

# Prevent Python from generating .pyc files (compiled bytecode files)
sys.dont_write_bytecode = True

# Import necessary modules and configuration settings
from prompts    import *
from model      import *
from utils      import *

printHeader(f"Enrich Concepts with Synonyms")

# Only proceed if formatted input data exists
exitIfFileNotExist(inputFileGeneration)

# Load the dataset from a pickle file
log(f"Loading data from \"{os.path.basename(inputFileGeneration)}\"...")
data    = pd.read_csv(inputFileGeneration)
log("Data loaded.")

hpoIDs  = list(set(data[hpoidColumn]))
if reduceToTestIDs:
    hpoIDs = testIDs
log(f"Identified {len(hpoIDs)} HPO concepts.")

parents     = {}
children    = {}

with newProgress() as progress:

    task    = newTask(progress, len(set(data[hpoidColumn])), 
        "Get Parents and Children")
    
    for hpoID in hpoIDs:
        children[hpoID] = getChildLabels(data, hpoID)
        parents[hpoID]  = getParentLabels(data, hpoID)
        progress.update(task, advance=1)

    progress.refresh()

log("Adding Definitions...")

logging.getLogger("vllm").setLevel(logging.ERROR)

log(f"Set up the LLM (\"{model_id}\")...")
model = Model(model = model_id)

definitions = data.loc[
    data[classColumn] == definitionClass, 
    hpoidColumn].tolist()

log(f"Found {len(definitions)} / {len(hpoIDs)} definitions.")

if (len(definitions) < len(hpoIDs)):
    noDefinitions = hpoIDs.copy()

    for hpoID in definitions:
        noDefinitions.remove(hpoID)

    log(f"Generating {len(noDefinitions)} missing definitions...")

    messages = []

    for hpoID in noDefinitions:
        messages.append(getPreTaskPart1(
            "".join(getElements(data, hpoID, labelClass))
        ))

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

        progress.refresh()

    formattedDefinition = pd.DataFrame({
        contentColumn   : definitionTexts,
        classColumn     : [enrichedSourceDefinitionClass] 
            * len(definitionTexts),
        hpoidColumn     : noDefinitions,
        typeColumn      : [""]                              
            * len(definitionTexts),
        systemColumn    : [model_name]                      
            * len(definitionTexts),
        roundColumn     : [-1]                              
            * len(definitionTexts)
    })

    data = pd.concat([data, formattedDefinition]).reset_index(drop = True)

    model.logPrompts()
    model.reset()
else:
    log("No definition generation required.")

log("Definition adding completed.")

messages = []

with newProgress() as progress:

    task = newTask(progress, len(hpoIDs) * generateTimes, "Creating Prompts")

    for hpoID in hpoIDs:
        for _ in range(0, generateTimes):
            messages.append(getAlternativeComplexPrompt1(
                "".join(getElements(data, hpoID, [labelClass])),
                "".join(getElements(data, hpoID, [definitionClass, 
                    enrichedSourceDefinitionClass])),
                "".join(getElements(data, hpoID, [commentClass])),
                parents[hpoID],
                children[hpoID]
            ))
            progress.update(task, advance=1)

    progress.refresh()

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
    
    for hpoID in hpoIDs:

        index = hpoIDs.index(hpoID)
        for t in range(0, generateTimes):
            generatedTextHPOid[index * generateTimes + t]   = \
                hpoID
            messagesHistory                                 = \
                messagesHistories[index * generateTimes + t]
            generatedText[index * generateTimes + t]        = \
                str(messagesHistory[-1][messageTextElement])
            generatedRound[index * generateTimes + t]       = \
                t + 1

            progress.update(task, advance=1)

    progress.refresh()

result = pd.DataFrame({
    contentColumn   : generatedText, 
    hpoidColumn     : generatedTextHPOid, 
    roundColumn     : generatedRound,
    classColumn     : [enrichedSourceExactSynonymClass] * len(generatedRound),
    systemColumn    : [model_id]                        * len(generatedRound),
    typeColumn      : [""]                              * len(generatedRound)
})
result.to_csv(outputFileGeneration, index = False)

if not os.path.exists(outputFileGenerationGold):
    log("Creating gold standard file...")
    
    gold = data[((
            data[classColumn] == labelClass
        ) | (
            data[classColumn].isin(synonymClasses)
        ))].drop_duplicates(ignore_index = True).reset_index(drop = True)
    gold.to_csv(outputFileGenerationGold)
    
    log("Created gold standard file.")
else:
    log("Gold standard file found, nothing to create.")

printHeader("Generation of Synonyms completed.")