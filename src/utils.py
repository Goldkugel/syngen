import pandas as pd
import sys
import os
from datetime import datetime

import Levenshtein  as lev

sys.dont_write_bytecode = True

from config import * 

from rich.table import Table
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TaskID, \
    TaskProgressColumn, TimeElapsedColumn

def cleanUp(data : pd.DataFrame) -> pd.DataFrame:
    hpoIDs = list(set(data[hpoidColumn].tolist()))
    datasets = []

    for hpoID in hpoIDs:
        subset = data[data[hpoidColumn] == hpoID].copy().reset_index(drop=True)
        remove = []

        for index, row in subset.iterrows():
            for index2, row2 in subset.iterrows():
                # If the rows are distinct but have the same content, class, 
                # language, and sytem they can be merged.
                if (index2 > index and
                    str(row[contentColumn]).lower()  == str(row2[contentColumn]).lower() and 
                    row[classColumn]    == row2[classColumn] and 
                    row[languageColumn] == row2[languageColumn] and 
                    row[systemColumn]   == row2[systemColumn]):
                    subset.loc[index, countColumn] = \
                        subset.loc[index, countColumn] + \
                            subset.loc[index2, countColumn]
                    remove.append(index2)

        subset = subset.drop(index=remove)
        datasets.append(subset)

    return pd.concat(datasets).reset_index(drop=True)

def formatting(string : str = "", quotationChar : str = quotationCharacter) -> list:
    ret = None

    if string is not None:
        parts = None
        if quotationChar in string:
            string = string[string.index(quotationCharacter):len(string)]
            parts = string.split(quotationChar)

            correct = -1
            index = 0
            waiting = False

            while (correct < index and index - correct <= 2 and index < len(parts)):
                part = parts[index]

                if part in [",", ", ", ""]:
                    parts[index] = ""
                    correct += 1
                    if waiting:
                        correct += 1
                        waiting = False
                else:
                    waiting = True
                        
                index += 1

            if waiting:
                correct += 1

            ret = parts[0:correct+1]
        else:
            ret = []

    return ret

def replaceQuotes(text : str = "", quotationChar : str = quotationCharacter):
    return text.replace("'", quotationChar). \
        replace("’", quotationChar). \
        replace("‘", quotationChar). \
        replace("\"", quotationChar)

# Function to retrieve labels of child concepts
def getChildLabels(
    data: pd.DataFrame, 
    hpoID: str
) -> list:
    """
    Get all labels of child concepts for a given ID in the specified language.
    """
    ret = []
    childIDs = getElements(data, hpoID, childrenClass, universalLanguage)
    for childID in childIDs:
        ret += getElements(data, childID, labelClass)
    return ret

# Function to retrieve labels of parent concepts
def getParentLabels(
    data: pd.DataFrame, 
    hpoID: str
) -> list:
    """
    Get all labels of parent concepts for a given ID in the specified language.
    """
    ret = []
    contentFilter = data[data[contentColumn] == hpoID]
    classFilter = contentFilter[contentFilter[classColumn] == childrenClass]
    parentIDs = classFilter[hpoidColumn].tolist()
    for parentID in parentIDs:
        ret += getElements(data, parentID, labelClass, "en")
    return ret

def getRows(
    data: pd.DataFrame, 
    hpoID: str,  
    # Default to label class
    className = labelClass,
    language: str = "en"
) -> pd.DataFrame:
    # Filter data to only include rows with the given ID.
    ret = data[data[hpoidColumn] == hpoID]

    if len(ret.index) > 0:
        ret = ret[ret[languageColumn] == language]

        if isinstance(className, str):
            ret = ret[ret[classColumn] == className]
        else:
            ret = ret[ret[classColumn].isin(className)]
    else:
        ret = None
        
    return ret

# Gets all elements that are stored in the given data frame, have a content that
# is written in the given language, have the given ID, and belong to the given 
# className.
# If it is known that only one element is available, a list with a single 
# element is returned; otherwise, a list with all matching elements is returned.
def getElements(
    data: pd.DataFrame, 
    hpoID: str,   
    # Default to label class
    className = labelClass,
    language: str = "en"
) -> list:
    ret = getRows(data, hpoID, className, language) 

    # Check if any elements remain after filtering.
    if ret is not None and len(ret.index) > 0:
        ret = ret[contentColumn].tolist()
    else:
        ret = []

    return ret


def isFile(file : str = "") -> bool:
    return os.path.isfile(file)

def getNextElementID() -> int:
    getNextElementID.lastElementID += 1
    return getNextElementID.lastElementID

def setNextElementID(data : pd.DataFrame) -> None:
    getNextElementID.lastElementID = max(data[elementIDColumn].values.tolist())

getNextElementID.lastElementID = 0

def newProgress() -> Progress:
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(complete_style=progressBarColor),
        TaskProgressColumn(),
        #TextColumn("ETA:"),
        #TimeRemainingColumn(),
        TextColumn("ET:"),
        TimeElapsedColumn(),
        TextColumn("Elem.s: {task.completed}/{task.total}"),
    )

def newTask(progress : Progress, iterations : int, text : str = "Taks") -> TaskID:
    return progress.add_task(
        text.ljust(progressBarTextLength), 
        total=iterations
    ) 

def printHeader(text : str = "") -> None:
    log(headerSeparator)
    log(headerChar + text.center(headerLen - 2) + headerChar)
    log(headerSeparator)

def printProcessing(file : str = "") -> None:
    log("Processing file '" + os.path.basename(file) + "'.")

def printProcessingDone(file : str = "") -> None:
    log("Processing file '" + os.path.basename(file) + "' completed.")

def printRowCount(data : pd.DataFrame = None) -> None:
    log("Row count: " + str(len(data.index)))

def printRead(file : str = "") -> None:
    log("Reading file '" + os.path.basename(file) + "'.")

def printWrite(file : str = "") -> None:
    log("Writing file '" + os.path.basename(file) + "'.")

def printReadDone(file : str = "") -> None:
    log("Reading file '" + os.path.basename(file) + "' completed.")

def printWriteDone(file : str = "") -> None:
    log("Writing file '" + os.path.basename(file) + "' completed.")

def readPickle(file : str = "") -> pd.DataFrame:
    printRead(file)
    ret = pd.read_pickle(file)
    printReadDone(file)
    return ret

def log(string : str, cmdline : bool = True, file : str = logFile) -> None:
    myfile = open(file=file, mode="a")
    string = "[" + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "] " + \
        string
    if cmdline:
        print(string)
    myfile.write(string + "\n")

def readPickle(file : str = "") -> pd.DataFrame:
    printRead(file)
    ret = pd.read_pickle(file)
    printReadDone(file)
    return ret

def writePickle(data : pd.DataFrame, file : str = "") -> None:
    printWrite(file)
    pd.to_pickle(data, file)
    printWriteDone(file)


def printDataSummary(data : pd.DataFrame = None) -> None:
    console = Console()

    # Create a table to display the summary.
    table = Table(title="Data Summary Report")

    # Add columns to the table.
    table.add_column("Description")
    table.add_column("Abs")
    table.add_column("Rel")

    # Add the value counts for the specified column to the table.
    value_counts = data[classColumn].value_counts()
    for value, count in value_counts.items():
        relativeAmount = round(count * 100 / len(data.index))
        table.add_row(str(value), str(count).rjust(len(str(len(data.index)))), 
            ("~" + str(relativeAmount) + "%").rjust(5))

    # Print the table.
    console.print(table)

def removeEmptyRows(data : pd.DataFrame) -> tuple[pd.DataFrame, int]:
    ret = data.copy()

    remove = []
    for index, row in ret.iterrows():
        if (len(str(row[contentColumn])) == 0):
            remove.append(index)

    ret = ret.drop(index=remove)

    ret = ret.reset_index(drop=True)

    return ret, len(remove)

def cleanUp(data : pd.DataFrame) -> tuple[pd.DataFrame, int]:
    hpoIDs = list(set(data[hpoidColumn].tolist()))
    datasets = []

    for hpoID in hpoIDs:
        subset = data[data[hpoidColumn] == hpoID].copy().reset_index(drop=True)
        remove = []

        for index, row in subset.iterrows():
            for index2, row2 in subset.iterrows():
                # If the rows are distinct but have the same content, class, 
                # language, and sytem they can be merged.
                if (index2 > index and
                    str(row[contentColumn]).lower()  == str(row2[contentColumn]).lower() and 
                    row[classColumn]    == row2[classColumn] and 
                    row[languageColumn] == row2[languageColumn] and 
                    row[systemColumn]   == row2[systemColumn]):
                    subset.loc[index, countColumn] = \
                        subset.loc[index, countColumn] + \
                            subset.loc[index2, countColumn]
                    remove.append(index2)

        subset = subset.drop(index=remove)
        datasets.append(subset)

    return pd.concat(datasets).reset_index(drop=True), len(remove)