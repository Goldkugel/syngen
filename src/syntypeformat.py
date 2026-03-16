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
exitIfFileNotExist(inputFileClassificationTypeFormatted)

# Load the dataset from a pickle file
classified    = readCSV(inputFileClassificationTypeFormatted)






with newProgress() as progress:

    task = newTask(progress, len(classified.index), "Formatting Answers")

    for index in range(0, len(classified.index)):
        classified.loc[index, answerColumn] = \
            formatAnswerClassificationType(str(classified[answerColumn][index]))
        progress.advance(task, advance = 1)

    progress.refresh()

writeCSV(classified, outputFileClassificationTypeFormatted)

log("Logging incorrect classified Synonyms...")

gold        = readCSV(inputFileClassificationType)
labels      = gold[gold[classColumn] == labelClass].copy().reset_index(drop = True)
count       = 0

for index, row in classified.iterrows():
    # It should log, when:
    # - The answer could not be formatted e.g. is undefined.
    # - No type in the gold dataset means "expert", therefore having an empty
    #       type column and an answer that is not "expert" means wrong
    #       classification.
    # - If the type is direct, it just means that the source of the synonym is
    #       found in the class, not in the axioms, and therefore the type in 
    #       gold dataset is "expert". If the answer is not "expert" means wrong
    #       classification. 
    # - If the answer or the gold class is "layperson" but the other is not.
    # This way all "layperson" and "expert" terms classified the wrong way are
    # being logged.
    #
    if (str(row[answerColumn]).lower() == undefinedSynonymType.lower() or
        (str(row[answerColumn]).lower() != expertSynonymType and row[typeColumn] == "") or
        (str(row[answerColumn]).lower() != expertSynonymType and row[typeColumn] == directSynonymType) or
        ((str(row[answerColumn]).lower() != row[typeColumn] and row[typeColumn] == laypersonSynonymType) and (str(row[answerColumn]).lower() == laypersonSynonymType or row[typeColumn] == laypersonSynonymType))):
        log(f"Label: \"{', '.join(getElements(labels, row[hpoidColumn], labelClass))}\", Synonym: \"{row[contentColumn]}\", Correct: \"{row[typeColumn]}\", Classified: \"{row[answerColumn]}\"", cmdline = False)
        count = count + 1

log(f"{count} incorrect classified Synonyms logged.")






minutes         = int((time.time() - start_time) // 60)

# Print a formatted header indicating the end of this processing stage
printHeader(f"Formatting completed [Minutes: {minutes}]")