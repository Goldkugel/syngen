import sys
import logging
import time

# Prevent Python from generating .pyc files (compiled bytecode files)
sys.dont_write_bytecode = True

# Import necessary modules and configuration settings
from prompts    import *
from utils      import *
from model      import *
from config     import *

logging.getLogger("vllm").setLevel(logging.ERROR)

printHeader(f"Classifying the Synonyms")
start_time = time.time()

# Only proceed if formatted input data exists
exitIfFileNotExist(inputFileClassificationType)

# Load the dataset from a pickle file
gold    = readCSV(inputFileClassificationType)






synonyms = gold[gold[classColumn].isin(synonymClasses)].copy().reset_index(drop = True)
hpoIDs = getHPOIDs(synonyms)
parents = {}
children = {}

with newProgress() as progress:
    
    task = newTask(progress, len(hpoIDs), "Get Parents and Children")

    for hpoID in hpoIDs:
        children[hpoID] = getChildLabels(gold, hpoID)
        parents[hpoID]  = getParentLabels(gold, hpoID)
        progress.update(task, advance=1)
    
    progress.refresh()

log(f"Set up the LLM ({model_id})...")
model = Model(model=model_id)

messages = []

synonyms = synonyms[synonyms[hpoidColumn].isin(hpoIDs)].copy().reset_index(drop = True)

with newProgress() as progress:

    task = newTask(progress, len(synonyms.index), "Set up Prompt(s)")

    for index, row in synonyms.iterrows():
        hpoID = row[hpoidColumn]

        messages.append(getSynonymClassPrompt(
            "".join(getElements(gold, hpoID, labelClass)),
            "".join(getElements(gold, hpoID, definitionClass)),
            "".join(getElements(gold, hpoID, commentClass)),
            parents[hpoID],
            children[hpoID],
            row[contentColumn]
        ))

        progress.update(task, advance=1)

    progress.refresh()

addedPrompts = model.addPrompt(userRole, messages)
log(f"{addedPrompts} prompts added. Start generating responses...")
model.generate()
model.logPrompts()

histories = model.getMessageHistories().copy()

synonyms[answerColumn] = [""] * len(synonyms.index)
for index, history in enumerate(histories):
    if history is not None and isinstance(history, list) and messageTextElement in history[-1].keys() and history[-1][messageTextElement] is not None:
        synonyms.loc[index, answerColumn] = str(history[-1][messageTextElement]).strip()
        synonyms.loc[index, systemColumn] = model_name






writeCSV(synonyms, outputFileClassification)

minutes         = int((time.time() - start_time) // 60)

# Print a formatted header indicating the end of this processing stage
printHeader(f"Synonyms Classified [Minutes: {minutes}]")