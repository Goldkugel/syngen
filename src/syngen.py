import pandas as pd
import sys

sys.path.insert(1, '../')

# Prevent Python from generating .pyc files (compiled bytecode files)
sys.dont_write_bytecode = True

# Import necessary modules and configuration settings
from prompts import *
from model import *
from hpot.src.createSynonymsChallenge.configSynonymsChallenge import *
from utils import *

printHeader(f"Enrich Concepts with Synonyms")

# Load the dataset from a pickle file
data = None
if useOutputAsInput and isFile(outputFile):
    data = readPickle(outputFile)
else:
    data = readPickle(inputFile)
    data[systemColumn] = [""] * len(data.index)
    data[countColumn] = [1] * len(data.index)

log("Removing empty data.")
data, lostByFilter = removeEmptyRows(data=data)
log(f"Removed {lostByFilter} entries.")

# Retrieve all unique concept IDs
hpoIDs = list(set(data[hpoidColumn].tolist()))

log("Set up the LLM.")
model = Model(model=model_id)

instructions = [getSystem()] * len(hpoIDs)
log(f"{model.addPrompt(systemRole, instructions)} system instructions " \
    "added to the model.")

# Add first instruction.
messages = []

for hpoID in hpoIDs:
    messages.append(getTaskPart1(
        label = "".join(getElements(data, hpoID, sourceLanguageShort, 
            [labelClass]
        )),
        definition = "".join(getElements(data, hpoID, sourceLanguageShort, 
            [definitionClass, enrichedSourceDefinitionClass]
        ))
    ))

log(f"{model.addPrompt(userRole, messages)} prompts added to " \
    "the model. Start generating responses.")
model.generate()

log(f"{model.addPrompt(userRole, [getTaskPart2()])} prompts " \
    "added to the model. Start generating responses.")
model.generate()

log(f"{model.addPrompt(userRole, [getTaskPart3()])} prompts " \
    "added to the model. Start generating responses.")
model.generate()

messages = []
for hpoID in hpoIDs:
    messages.append(getTaskPart4(
        getParentLabels(data, hpoID, sourceLanguageShort)
    ))

c = model.addPrompt(userRole, messages)
log(f"{c} prompts added to the model. Start generating responses.")
model.generate()

messages = []
for hpoID in hpoIDs:
    messages.append(getTaskPart5(
        getChildLabels(data, hpoID, sourceLanguageShort)
    ))

c = model.addPrompt(userRole, messages)
log(f"{c} prompts added to the model. Start generating responses.")
model.generate()

log(f"{model.addPrompt(userRole, [getTaskPart6()])} prompts " \
    "added to the model. Start generating responses.")
model.generate()

synonymLists = []
messagesHistories = model.getMessageHistories()

incorrectFormat = []

# Process each concept ID to enrich synonyms
with newProgress() as progress:
    task = newTask(progress, len(messagesHistories), "Processing Synonyms")
    
    for index, messagesHistory in enumerate(messagesHistories):
        
        answer = str(messagesHistory[-1][messageTextElement]).strip()
        answer = replaceQuotes(answer.replace("\n", "").replace(".", ""))

        synonymList = list(set(formatting(answer)))
        
        if len(synonymList) == 0:
            if len(answer) > 0:
                incorrectFormat.append(hpoIDs[index])
        else:
            source = data[
                (data[hpoidColumn] == hpoIDs[index]) & 
                (data[languageColumn] == sourceLanguageShort) & 
                (data[classColumn] == labelClass)
            ]
            elementIDs = source[elementIDColumn].tolist()

            id = -1
            if len(elementIDs) == 1:
                id = elementIDs[0]

            synonymLists.append(pd.DataFrame({
                languageColumn  : [sourceLanguageShort] * len(synonymList),
                contentColumn   : synonymList,
                classColumn     : [enrichedSourceExactSynonymClass] * 
                    len(synonymList),
                hpoidColumn     : [hpoIDs[index]] * len(synonymList),
                sourceElemement : [id] * len(synonymList),
                systemColumn    : [model_id] * len(synonymList),
                countColumn     : [1] * len(synonymList),
                elementIDColumn : 
                    [getNextElementID() for _ in range(0, len(synonymList))]
            }))
        progress.update(task, advance=1)

log(f"{len(incorrectFormat)} not formatted Answers.")
if len(incorrectFormat) > 0:
    for inc in incorrectFormat:
        log(f"Response for {inc} was not formatted.")

log("Merging data.")
synonymLists.append(data)
data = pd.concat(synonymLists)
log("Merged data.")

log("Removing empty data.")
data, lostByFilter = removeEmptyRows(data=data)
log(f"Removed {lostByFilter} entries.")

# Remove duplicate rows
log("Merged duplicates.")
data, lostByFilter = cleanUp(data)
log(f"Merged {lostByFilter} duplicates.")

# Reset index after cleaning
data = data.reset_index(drop=True)

model.logPrompts()

# Display summary of the dataset
printRowCount(data)
printDataSummary(data)

# Save the updated dataset
writePickle(data, outputFile)
data.to_csv("/home/pallaoro/hpot/data/createSynonymsChallenge/output.csv")

printHeader(f"Enrichment of {sourceLanguage} Terms with Synonyms completed")