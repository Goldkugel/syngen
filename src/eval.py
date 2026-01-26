import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import seaborn as sns
import sys
import math
import random

# Prevent Python from generating .pyc files (compiled bytecode files)
sys.dont_write_bytecode = True

# Import necessary modules and configuration settings
from prompts import *
from model import *
from utils import *
from config     import *

printHeader("Evaluating the Results")

printRead(f"Reading \"{os.path.basename(inputFileEvaluatedTerms)}\"...")
generated = pd.read_csv(inputFileEvaluatedTerms).reset_index(drop=True)
generated.loc[contentColumn] = generated[hpoidColumn] + " " + generated[contentColumn]
printRead(f"Read \"{inputFileNameEvaluatedTerms}\".")

printRead(f"Reading \"{os.path.basename(inputFileEvaluatedGold)}\"...")
gold = pd.read_csv(inputFileEvaluatedGold).reset_index(drop=True)
printRead(f"Read \"{os.path.basename(inputFileEvaluatedGold)}\"...")

labels = list(set(gold[gold[classColumn] == labelClass][contentColumn]))
goldExactSynonymData = gold[gold[classColumn] == exactSynonymClass]
goldData = goldExactSynonymData[~goldExactSynonymData[contentColumn].isin(labels)].copy()
goldData[contentColumn] = goldData[hpoidColumn] + " " + goldData[contentColumn]

generated[systemColumn] = generated[systemColumn].fillna("")
generated.loc[generated[systemColumn] == "", systemColumn] = ""

systems = list(set(generated[systemColumn].tolist()))
systemsName = []
for system in systems:
    if len(system) > 0:
        systemsName.append(system[str(system).index("/") + 1:])

log(f"Found Systems: \"{'\", \"'.join(systemsName)}\"")
generated = generated[generated[systemColumn] != ""]
systems = list(set(generated[systemColumn].tolist()))

colors = plt.cm.tab10(range(len(systems) + 1))

roundCount = int(max(generated[roundColumn].tolist()))

xRoundTicks = 5
xRound = np.arange(1, roundCount + 1)
xRoundTickList = [1] + list(np.arange(xRoundTicks, roundCount + 1, xRoundTicks))

def getSystemData(data : pd.DataFrame, system : str) -> list:
    return list(set(data[data[systemColumn] == system].tolist()))

def getSystemRoundData(data : pd.DataFrame, system : str, round : int) -> list:
    # Reduce Data to System data
    systemData = data[data[systemColumn] == system]
    # Reduce Data to Round data
    systemRoundData = systemData[systemData[roundColumn] == round]
    
    return list(set(systemRoundData[contentColumn].tolist()))

def getCumulativeSystemRoundData(data : pd.DataFrame, system : str, round : int) -> list:
    # Reduce Data to System data
    systemData = data[data[systemColumn] == system]
    # Reduce Data to Round and pre-Round data
    systemRoundData = systemData[systemData[roundColumn] <= round]
    
    return list(set(systemRoundData[contentColumn].tolist()))

def getSystemRoundHPOData(data : pd.DataFrame, system : str, round : int, hpoID : str) -> list:
    # Reduce Data to System data
    systemData = data[data[systemColumn] == system]
    # Reduce Data to Round data
    systemRoundData = systemData[systemData[roundColumn] == round]
    # Reduce Data to HPO ID data
    systemRoundHPOData = systemRoundData[systemRoundData[hpoidColumn] == hpoID]
    
    return list(set(systemRoundHPOData[contentColumn].tolist()))

def getCumulativeSystemRoundHPOData(data : pd.DataFrame, system : str, round : int, hpoID : str) -> list:
    # Reduce Data to System data
    systemData = data[data[systemColumn] == system]
    # Reduce Data to Round data
    systemRoundData = systemData[systemData[roundColumn] <= round]
    # Reduce Data to HPO ID data
    systemRoundHPOData = systemRoundData[systemRoundData[hpoidColumn] == hpoID]
    
    return list(set(systemRoundHPOData[contentColumn].tolist()))

def getHighestAppearance(data : pd.DataFrame, system : str) -> int:
    element = max(set(data.loc[data[systemColumn] == system, contentColumn].tolist()), key=data.loc[data[systemColumn] == system, contentColumn].tolist().count)
    return len(generated[(generated[contentColumn] == element) & (generated[systemColumn] == system)].index)

def getAppearanceSystemHPOData(data : pd.DataFrame, system : str, hpoID : str, count : int) -> list:
    ret = []
    
    # Reduce Data to System data
    systemData = data[data[systemColumn] == system]
    # Reduce Data to HPO ID data
    systemHPOData = systemData[systemData[hpoidColumn] == hpoID]

    gen = systemHPOData[contentColumn].tolist()
    for g in list(set(gen)):
        c = gen.count(g)
        if c >= count:
            ret.append(g)

    return ret

def getNewSynonymsSystemRound(data : pd.DataFrame, system : str, round : int) -> list:
    ret = []

    systemData = data[data[systemColumn] == system]
    # Reduce Data to Round data

    if round == 0:
        ret = list(set(systemData[systemData[roundColumn] == 0][contentColumn]))
    else:
        systemRoundData = systemData[systemData[roundColumn] <= round][contentColumn].tolist()
        systemPreviousRoundData = systemData[systemData[roundColumn] <= round - 1][contentColumn].tolist()

        ret = list(set(systemRoundData).difference(set(systemPreviousRoundData)))

    return ret

def splitEqualParts(start : float, end : float, parts : int, cut : int = 2) -> list:
    ret = list(np.linspace(start, end, parts))
    for index in range(0, len(ret)):
        ret[index] = math.trunc(ret[index] * math.pow(10, cut)) / math.pow(10, cut)
    return ret

def splitEqualPartsLabels(start : float, end : float, parts : int, cut : int = 2, prefix : str = "", postfix : str = "", multiplier : int = 1) -> list:
    ret = splitEqualParts(start, end, parts, cut)

    for index in range(0, len(ret)):

        value = str(ret[index] * multiplier)

        if "." in value:
            value = value.split(".")
            
            if cut > 0:
                if len(value[1]) > cut:
                    tmp = value[1]
                    value[1] = str(round(float(tmp[0:cut])))
            else:
                value[1] = "0"

            if not value[1] == "0":
                value = ".".join(value)
            else:
                value = value[0]

        ret[index] = prefix + str(value) + postfix

    return ret

def getPercentage(interval: float = 0.1, additional: int = 1) -> list:
    ret = []
    numbers = list(np.arange(0, 1.01, interval))

    for number in numbers:
        ret.append(str(math.floor(100 * number)) + "%")

    for _ in range(0, additional):
        ret.append("")

    return ret

log("Creating Plot 'Overall Amount of Distinct Generated Synonyms per Round'...")

result = {}
maxVal = 0

for system in systems:
    count = [0] * roundCount

    for r in range(0, roundCount):
        count[r] = len(getSystemRoundData(generated, system, r + 1))
        if count[r] > maxVal:
            maxVal = count[r]

    result[system] = {
        "counts" : count 
    }

yticks = 10

# Extract data for plotting
data = [result[system]["counts"] for system in systems]

# Create the boxplot
plt.figure(figsize=(3 * len(systems), 5))
plt.boxplot(
    data,
    showfliers=True,
    widths=[0.6] * len(systems),
    patch_artist=True,
    boxprops=dict(facecolor="aliceblue", color="cornflowerblue"),
    medianprops=dict(color="cornflowerblue"),
    whiskerprops=dict(color="cornflowerblue"),
    capprops=dict(color="cornflowerblue"),
    flierprops=dict(markeredgecolor="cornflowerblue")
)
plt.title("Overall Amount of Distinct Generated Synonyms per Round")
plt.suptitle("")
plt.xlabel("")
plt.xticks(range(1, len(systems) + 1), labels=systemsName)
plt.ylabel("Amount of Distinct Generated Synonyms per Round")
plt.yticks(range(0, int(maxVal * 1.1) , int(maxVal * 1.1 / yticks)))
plt.tight_layout()
plt.grid(visible=True)
plt.savefig(outputFileDistinctSynonymsRound, dpi=300, bbox_inches="tight")
log("Creaed Plot 'Overall Amount of Distinct Generated Synonyms per Round'.")

log("Creating Plot 'Overall Amount of Distinct Generated Synonyms per Concept'...")
result = {}
maxVal = 0

for system in systems:
    count = [0] * len(testIDs)

    for index, hpoID in enumerate(testIDs):
        count[index] = len(getCumulativeSystemRoundHPOData(generated, system, roundCount + 1, hpoID))
        if count[index] > maxVal:
            maxVal = count[index]

    result[system] = {
        "counts" : count 
    }

yticks = math.ceil(maxVal / 500)

# Extract data for plotting
data = [result[system]["counts"] for system in systems]

# Create the boxplot
plt.figure(figsize=(3 * len(systems), 5))
plt.boxplot(
    data,
    showfliers=True,
    widths=[0.6] * len(systems),
    patch_artist=True,
    boxprops=dict(facecolor="aliceblue", color="cornflowerblue"),
    medianprops=dict(color="cornflowerblue"),
    whiskerprops=dict(color="cornflowerblue"),
    capprops=dict(color="cornflowerblue"),
    flierprops=dict(markeredgecolor="cornflowerblue")
)
plt.title("Overall Amount of Distinct Generated Synonyms per Concept")
plt.suptitle("")
plt.xlabel("")
plt.xticks(range(1, len(systems) + 1), labels=systemsName)
plt.ylabel("Amount of Distinct Generated Synonyms per Concept")

yticksList = splitEqualParts(0, maxVal + 1, 10, 0)
yticksLabels = splitEqualPartsLabels(0, maxVal + 1, 10, 0)
plt.yticks(
    ticks=yticksList,
    labels=yticksLabels
)

plt.tight_layout()
plt.grid(visible=True)
plt.savefig(outputFileDistinctSynonymsConcept, dpi=300, bbox_inches="tight")
log("Created Plot 'Overall Amount of Distinct Generated Synonyms per Concept'.")

log("Creating Plot 'Relative Amount of new generated Synonyms'...")
result = {}
maxVal = 0

for index, system in enumerate(systems):
    counts = [0] * roundCount
    relatives = [0] * roundCount
    
    for r in range(0, roundCount):
        counts[r] = len(getNewSynonymsSystemRound(generated, system, r + 1))
        relatives[r] = counts[r] / len(getSystemRoundData(generated, system, r + 1))

        if counts[r] > maxVal:
            maxVal = counts[r]

    result[system] = {
        "counts" : counts,
        "relatives" : relatives
    }

_, axs = plt.subplots(1, 2, figsize=(12, 5))

yticks = math.ceil(maxVal / 5)

for system in result.keys():
    axs[0].plot(xRound, result[system]["relatives"], marker='o', color=colors[systems.index(system)], linewidth=2, label=systemsName[systems.index(system)])
    axs[1].plot(xRound, result[system]["counts"], marker='s', color=colors[systems.index(system)], linewidth=2, label=systemsName[systems.index(system)])

axs[0].set_xlabel("Round")
axs[0].set_ylabel("Newly generated Synonyms (relative)")
axs[0].grid(True)
axs[0].set_yticks(splitEqualParts(0, 1, 10, 2))
axs[0].set_yticklabels(splitEqualPartsLabels(0, 1, 10, 2, "", "%", 100))
axs[0].set_xticks(xRoundTickList)
axs[0].set_title("Relative Amount of new generated Synonyms")

axs[1].set_yticks(splitEqualParts(0, maxVal + 1, 10, 0)) 
axs[1].set_yticklabels(splitEqualPartsLabels(0, maxVal + 1, 10, 0))
axs[1].set_ylabel("Newly generated Synonyms (absolute)")
axs[1].set_xticks(xRoundTickList)
axs[1].set_xlabel("Round")
axs[1].grid(True)
axs[1].set_title("Absolute Amount of new generated Synonyms")

# Combine legends from both axes
lines_1, labels_1 = axs[0].get_legend_handles_labels()
lines_2, labels_2 = axs[1].get_legend_handles_labels()
axs[1].legend(lines_1 + lines_2, labels_1 + labels_2)

#plt.title("Newly generated Synonyms per Round")
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.savefig(outputFileNewSynonymsRound, dpi=300, bbox_inches="tight")
log("Created Plot 'Relative Amount of new generated Synonyms'.")

log("Creating Plot 'Cumulative Recall and Precision'...")
maxprecision = 0
maxrecall = 0

result = {}
goldSynonyms = set(str(element).lower() for element in goldData[contentColumn].tolist())

for index, system in enumerate(systems):

    coverage  = [0] * roundCount
    precision = [0] * roundCount
    recall    = [0] * roundCount

    for r in range(0, roundCount):
        
        generatedSynonyms = set(str(element).lower() for element in getCumulativeSystemRoundData(generated, system, r + 1))
        intersect = list(goldSynonyms.intersection(generatedSynonyms))
        
        coverage[r] = len(intersect)
        precision[r] = coverage[r] / len(generatedSynonyms)
        recall[r] = coverage[r] / len(goldData.index)

    if (maxrecall < max(recall)):
        maxrecall = max(recall)
    if (maxprecision) < max(precision):
        maxprecision = max(precision)

    result[system] = {
        "coverage" : coverage,
        "precision" : precision,
        "recall" : recall
    }

maxrecall = math.ceil(maxrecall * 100) / 100
maxprecision = math.ceil(maxprecision * 100) / 100

_, axs = plt.subplots(1, 2, figsize=(14, 4))

for system in result.keys():
    axs[0].plot(xRound, result[system]["precision"], marker='o', color=colors[systems.index(system)], linewidth=2, label=systemsName[systems.index(system)])
    axs[1].plot(xRound, result[system]["recall"], marker='x', color=colors[systems.index(system)], linewidth=2, label=systemsName[systems.index(system)])

axs[0].set_xlabel("Round")
axs[1].set_xlabel("Round")

axs[0].set_ylabel("Percentage")
axs[1].set_ylabel("Percentage")

axs[0].grid(True)
axs[1].grid(True)

yticks0 = splitEqualParts(0, maxprecision, 10, 3)
yticks1 = splitEqualParts(0, maxrecall, 10, 2)

axs[0].set_yticks(yticks0)
axs[1].set_yticks(yticks1)

yticks0 = splitEqualPartsLabels(0, maxprecision, 10, 3, "", "%", 100)
yticks1 = splitEqualPartsLabels(0, maxrecall, 10, 2, "", "%", 100)

axs[0].set_yticklabels(yticks0)
axs[1].set_yticklabels(yticks1)

axs[0].set_xticks(xRoundTickList)
axs[1].set_xticks(xRoundTickList)

axs[0].set_title("Cumulative Precision")
axs[1].set_title("Cumulative Recall")

plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.savefig(outputFileCumulativeRecallPrecision, dpi=300, bbox_inches="tight")
log("Created Plot 'Cumulative Recall and Precision'.")

log("Creating Plot 'Appearance based Recall, Precision, F1, and Count Plots'...")
maxprecision = 0
maxrecall = 0
maxAppearance = 0
maxf1 = 0
maxCoverage = 0

result = {}

goldSynonymsLower = set([str(s).lower() for s in goldSynonyms])

for _, system in enumerate(systems):

    print(system)

    generatedSynonyms = [str(s).lower() for s in generated.loc[generated[systemColumn] == system, contentColumn].tolist()]

    counts = {}
    for synonym in generatedSynonyms:
        if synonym in counts.keys():
            counts[synonym] = counts[synonym] + 1
        else:
            counts[synonym] = 1
            
    maxAppearanceSystem = max(list(counts.values()))
    print(maxAppearanceSystem)

    if maxAppearanceSystem > maxAppearance:
        maxAppearance = maxAppearanceSystem

    coverage    = [0] * maxAppearanceSystem
    precision   = [0] * maxAppearanceSystem
    recall      = [0] * maxAppearanceSystem
    f1          = [0] * maxAppearanceSystem
    appearances = [0] * maxAppearanceSystem

    for appearance in range(0, maxAppearanceSystem):

        appearances[appearance] = appearance

        generatedSynonymsAppearances = [s for s, c in counts.items() if c >= appearance + 1]

        if len(generatedSynonymsAppearances) > 0:
            intersect = list(goldSynonymsLower.intersection(generatedSynonymsAppearances))

            if len(intersect) > 0:
                coverage[appearance] = len(intersect)
                precision[appearance] = coverage[appearance] / len(generatedSynonymsAppearances)
                recall[appearance] = coverage[appearance] / len(goldData.index)
                f1[appearance] = (2 * precision[appearance] * recall[appearance]) / (recall[appearance] + precision[appearance])

    if (maxrecall < max(recall)):
        maxrecall = max(recall)
    if (maxprecision) < max(precision):
        maxprecision = max(precision)
    if maxCoverage < max(coverage):
        maxCoverage = max(coverage)
    if maxf1 < max(f1):
        maxf1 = max(f1)

    result[system] = {
        "coverage" : coverage,
        "precision" : precision,
        "recall" : recall,
        "f1" : f1, 
        "appearances" : appearances
    }

maxrecall = math.ceil(maxrecall * 100) / 100
maxprecision = math.ceil(maxprecision * 100) / 100

_, axs = plt.subplots(2, 2, figsize=(18, 10))
xRoundTicks = 10
xAppearanceTickList = [1] + list(np.arange(xRoundTicks, maxAppearance + 1, xRoundTicks))

for system in result.keys():
    axs[0][0].plot(result[system]["appearances"], result[system]["precision"], marker='o', color=colors[systems.index(system)], linewidth=2, label=systemsName[systems.index(system)])
    axs[0][1].plot(result[system]["appearances"], result[system]["recall"], marker='x', color=colors[systems.index(system)], linewidth=2, label=systemsName[systems.index(system)])
    axs[1][0].plot(result[system]["appearances"], result[system]["coverage"], marker='s', color=colors[systems.index(system)], linewidth=2, label=systemsName[systems.index(system)])
    axs[1][1].plot(result[system]["appearances"], result[system]["f1"], marker='s', color=colors[systems.index(system)], linewidth=2, label=systemsName[systems.index(system)])

axs[0][0].set_xlabel("Appearance")
axs[0][1].set_xlabel("Appearance")
axs[1][0].set_xlabel("Appearance")
axs[1][1].set_xlabel("Appearance")

axs[0][0].set_ylabel("Percentage")
axs[0][1].set_ylabel("Percentage")
axs[1][0].set_ylabel("Count")
axs[1][1].set_ylabel("F1-Score")

axs[0][0].grid(True)
axs[0][1].grid(True)
axs[1][0].grid(True)
axs[1][1].grid(True)

yticks0 = splitEqualParts(0, maxprecision, 10, 3)
yticks1 = splitEqualParts(0, maxrecall, 10, 2)
yticks2 = splitEqualParts(0, maxCoverage + 1, 10, 0)
yticks3 = splitEqualParts(0, maxf1, 10, 3)

axs[0][0].set_yticks(yticks0)
axs[0][1].set_yticks(yticks1)
axs[1][0].set_yticks(yticks2)
axs[1][1].set_yticks(yticks3)

yticks0 = splitEqualPartsLabels(0, maxprecision, 10, 3, "", "%", 100)
yticks1 = splitEqualPartsLabels(0, maxrecall, 10, 2, "", "%", 100)
yticks2 = splitEqualPartsLabels(0, maxCoverage + 1, 10, 0, "", "", 1)
yticks3 = splitEqualPartsLabels(0, maxf1, 10, 3, "", "", 1)

axs[0][0].set_yticklabels(yticks0)
axs[0][1].set_yticklabels(yticks1)
axs[1][0].set_yticklabels(yticks2)
axs[1][1].set_yticklabels(yticks3)

axs[0][0].set_xticks(xAppearanceTickList)
axs[0][1].set_xticks(xAppearanceTickList)
axs[1][0].set_xticks(xAppearanceTickList)
axs[1][1].set_xticks(xAppearanceTickList)

axs[0][0].set_title("Precision")
axs[0][1].set_title("Recall")
axs[1][0].set_title(f"Synonym Count")
axs[1][1].set_title("F1-Score")

plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.savefig(outputFileRecallPrecisionF1Appearance, dpi=300, bbox_inches="tight")
log("Created Plot 'Appearance based Recall, Precision, F1, and Count Plots'.")

log("Creating Plot 'Combined Recall and Precision'...")
maxprecision = 0
maxrecall = 0

result = {}
goldSynonyms = set(str(element).lower() for element in goldData[contentColumn].tolist())

system = "combined"

coverage  = [0] * roundCount
precision = [0] * roundCount
recall    = [0] * roundCount

for r in range(0, roundCount):
    
    # Reduce Data to Round and pre-Round data
    systemRoundData = generated[generated[roundColumn] <= r + 1]
    systemRoundData = list(set(systemRoundData[contentColumn].tolist()))

    generatedSynonyms = set(str(element).lower() for element in systemRoundData)
    intersect = list(goldSynonyms.intersection(generatedSynonyms))
    
    coverage[r] = len(intersect)
    precision[r] = coverage[r] / len(generatedSynonyms)
    recall[r] = coverage[r] / len(goldData.index)

if (maxrecall < max(recall)):
    maxrecall = max(recall)
if (maxprecision) < max(precision):
    maxprecision = max(precision)

result[system] = {
    "coverage" : coverage,
    "precision" : precision,
    "recall" : recall
}

maxrecall = math.ceil(maxrecall * 100) / 100
maxprecision = math.ceil(maxprecision * 100) / 100

_, axs = plt.subplots(1, 2, figsize=(14, 4))

axs[0].plot(xRound, result[system]["precision"], marker='o', linewidth=2, label="Combination")
axs[1].plot(xRound, result[system]["recall"], marker='x', linewidth=2, label="Combination")

axs[0].set_xlabel("Round")
axs[1].set_xlabel("Round")

axs[0].set_ylabel("Percentage")
axs[1].set_ylabel("Percentage")

axs[0].grid(True)
axs[1].grid(True)

yticks0 = splitEqualParts(0, maxprecision, 10, 3)
yticks1 = splitEqualParts(0, maxrecall, 10, 2)

axs[0].set_yticks(yticks0)
axs[1].set_yticks(yticks1)

yticks0 = splitEqualPartsLabels(0, maxprecision, 10, 3, "", "%", 100)
yticks1 = splitEqualPartsLabels(0, maxrecall, 10, 2, "", "%", 100)

axs[0].set_yticklabels(yticks0)
axs[1].set_yticklabels(yticks1)

axs[0].set_xticks(xRoundTickList)
axs[1].set_xticks(xRoundTickList)

axs[0].set_title("Cumulative Precision")
axs[1].set_title("Cumulative Recall")

plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.savefig(outputFileCumulativeRecallPrecisionTotal, dpi=300, bbox_inches="tight")
log("Created Plot 'Combined Recall and Precision'.")

log("Creating Plot 'Combined Recall, Precision, F1, and Count Plot'...")
maxprecision = 0
maxrecall = 0
maxAppearance = 0
maxf1 = 0
maxCoverage = 0

result = {}

goldSynonymsLower = set([str(s).lower() for s in goldSynonyms])
generatedSynonyms = [str(s).lower() for s in generated[contentColumn].tolist()]

counts = {}
for synonym in generatedSynonyms:
    if synonym in counts.keys():
        counts[synonym] = counts[synonym] + 1
    else:
        counts[synonym] = 1
        
maxAppearanceSystem = max(list(counts.values()))

if maxAppearanceSystem > maxAppearance:
    maxAppearance = maxAppearanceSystem

coverage    = [0] * maxAppearanceSystem
precision   = [0] * maxAppearanceSystem
recall      = [0] * maxAppearanceSystem
f1          = [0] * maxAppearanceSystem
appearances = [0] * maxAppearanceSystem

for appearance in range(0, maxAppearanceSystem):

    appearances[appearance] = appearance

    generatedSynonymsAppearances = [s for s, c in counts.items() if c >= appearance + 1]

    if len(generatedSynonymsAppearances) > 0:
        intersect = list(goldSynonymsLower.intersection(generatedSynonymsAppearances))

        if len(intersect) > 0:
            coverage[appearance] = len(intersect)
            precision[appearance] = coverage[appearance] / len(generatedSynonymsAppearances)
            recall[appearance] = coverage[appearance] / len(goldData.index)
            f1[appearance] = (2 * precision[appearance] * recall[appearance]) / (recall[appearance] + precision[appearance])

if (maxrecall < max(recall)):
    maxrecall = max(recall)
if (maxprecision) < max(precision):
    maxprecision = max(precision)
if maxCoverage < max(coverage):
    maxCoverage = max(coverage)
if maxf1 < max(f1):
    maxf1 = max(f1)

result[system] = {
    "coverage" : coverage,
    "precision" : precision,
    "recall" : recall,
    "f1" : f1, 
    "appearances" : appearances
}

maxrecall = math.ceil(maxrecall * 100) / 100
maxprecision = math.ceil(maxprecision * 100) / 100

_, axs = plt.subplots(2, 2, figsize=(18, 10))
xRoundTicks = 100
xAppearanceTickList = [1] + list(np.arange(xRoundTicks, maxAppearance + 1, xRoundTicks))

axs[0][0].plot(result[system]["appearances"], result[system]["precision"], marker='o', linewidth=2, label="Combination")
axs[0][1].plot(result[system]["appearances"], result[system]["recall"], marker='x', linewidth=2, label="Combination")
axs[1][0].plot(result[system]["appearances"], result[system]["coverage"], marker='s', linewidth=2, label="Combination")
axs[1][1].plot(result[system]["appearances"], result[system]["f1"], marker='s', linewidth=2, label="Combination")

axs[0][0].set_xlabel("Appearance")
axs[0][1].set_xlabel("Appearance")
axs[1][0].set_xlabel("Appearance")
axs[1][1].set_xlabel("Appearance")

axs[0][0].set_ylabel("Percentage")
axs[0][1].set_ylabel("Percentage")
axs[1][0].set_ylabel("Count")
axs[1][1].set_ylabel("F1-Score")

axs[0][0].grid(True)
axs[0][1].grid(True)
axs[1][0].grid(True)
axs[1][1].grid(True)

yticks0 = splitEqualParts(0, maxprecision, 10, 3)
yticks1 = splitEqualParts(0, maxrecall, 10, 2)
yticks2 = splitEqualParts(0, maxCoverage + 1, 10, 0)
yticks3 = splitEqualParts(0, maxf1, 10, 3)

axs[0][0].set_yticks(yticks0)
axs[0][1].set_yticks(yticks1)
axs[1][0].set_yticks(yticks2)
axs[1][1].set_yticks(yticks3)

yticks0 = splitEqualPartsLabels(0, maxprecision, 10, 3, "", "%", 100)
yticks1 = splitEqualPartsLabels(0, maxrecall, 10, 2, "", "%", 100)
yticks2 = splitEqualPartsLabels(0, maxCoverage + 1, 10, 0, "", "", 1)
yticks3 = splitEqualPartsLabels(0, maxf1, 10, 3, "", "", 1)

axs[0][0].set_yticklabels(yticks0)
axs[0][1].set_yticklabels(yticks1)
axs[1][0].set_yticklabels(yticks2)
axs[1][1].set_yticklabels(yticks3)

axs[0][0].set_xticks(xAppearanceTickList)
axs[0][1].set_xticks(xAppearanceTickList)
axs[1][0].set_xticks(xAppearanceTickList)
axs[1][1].set_xticks(xAppearanceTickList)

axs[0][0].set_title("Precision")
axs[0][1].set_title("Recall")
axs[1][0].set_title(f"Synonym Count")
axs[1][1].set_title("F1-Score")

plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.savefig(outputFileRecallPrecisionF1AppearanceTotal, dpi=300, bbox_inches="tight")
log("Created Plot 'Combined Recall, Precision, F1, and Count Plot'.")

printHeader("Evaluation completed.")