import sys
import time

# Prevent Python from generating .pyc files (compiled bytecode files)
sys.dont_write_bytecode = True

# Import necessary modules and configuration settings
from config     import *
from utils      import *

printHeader(f"Fomratting Classification of Synonyms")
start_time = time.time()

# Only proceed if formatted input data exists
exitIfFileNotExist(inputFileClassificationFormatted)

# Load the dataset from a pickle file
classified    = readCSV(inputFileClassificationFormatted)







classified[confidenceColumn] = [-1] * len(classified.index)

with newProgress() as progress:

    task = newTask(progress, len(classified.index), "Formatting Answers")
    for index in range(0, len(classified.index)):
        classified.loc[index, answerColumn], \
            classified.loc[index, confidenceColumn] = \
            formatAnswerClassification(str(classified[answerColumn][index]))
        progress.advance(task)

    progress.refresh()

log("Logging incorrect classified Synonyms...")

gold        = readCSV(inputFileClassification)
labels      = gold[gold[classColumn] == labelClass].copy().reset_index(drop = True)
count       = 0

for index, row in classified.iterrows():
    # It should log, when:
    # - The answer could not be formatted e.g. is undefined.
    # - If the answer or the gold class is "exact" or "related" but the other 
    #       is not equal.
    # This way all "exact" and "related" terms classified the wrong way are
    # being logged.
    #
    if (str(row[answerColumn]).lower() == undefinedSynonymType.lower() or
        (str(row[answerColumn]).lower() != str(row[classColumn]).lower()) and
            (str(row[answerColumn]).lower() in [exactSynonymClass, relatedSynonymClass] or
             str(row[classColumn]).lower() in [exactSynonymClass, relatedSynonymClass])):
        log(f"Label: \"{', '.join(getElements(labels, row[hpoidColumn], labelClass))}\", Synonym: \"{row[contentColumn]}\", Correct: \"{row[classColumn]}\", Classified: \"{row[answerColumn]}\"", cmdline = False)
        count = count + 1







print(len(classified.index))
writeCSV(classified, outputFileClassificationFormatted)

minutes         = int((time.time() - start_time) // 60)

printHeader(f"Formatting completed")