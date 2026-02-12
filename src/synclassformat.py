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

with newProgress() as progress:

    task = newTask(progress, len(classified.index), "Formatting Answers")
    for index in range(0, len(classified.index)):
        classified.loc[index, answerColumn] = \
            formatAnswerClassification(str(classified[answerColumn][index]))
        progress.advance(task)

    progress.refresh()

writeCSV(classified, outputFileClassificationFormatted)

end_time = time.time()
elapsed_seconds = end_time - start_time
minutes = int(elapsed_seconds // 60)

printHeader(f"Fomratting completed")