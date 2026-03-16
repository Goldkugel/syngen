import numpy                as np
import matplotlib.pyplot    as plt
import sys
import time

# Prevent Python from generating .pyc files (compiled bytecode files)
sys.dont_write_bytecode = True

# Import necessary modules and configuration settings
from model      import *
from utils      import *
from config     import *

from collections import Counter

printHeader("Evaluating the Results of Synonym Type Classification")
start_time = time.time()

# Only proceed if formatted input data exists
exitIfFileNotExist(inputFileClassificationTypeEvaluation)

classified_complete = readCSV(inputFileClassificationTypeEvaluation)






# Change Datatype to String. 
classified_complete[classColumn]    = classified_complete[classColumn ].str.lower()
classified_complete[answerColumn]   = classified_complete[answerColumn].str.lower()
classified_complete[typeColumn]     = classified_complete[typeColumn  ].str.lower()

classified_complete[typeColumn]     = classified_complete[typeColumn].replace(np.nan, expertSynonymType)
classificationClasses               = [expertSynonymType, laypersonSynonymType]
classified_complete                 = classified_complete[(classified_complete[typeColumn].isin(classificationClasses)) & (classified_complete[systemColumn] != "")].copy().reset_index(drop = True)

systemsName  = list(set(classified_complete[systemColumn].tolist()))
systemsName.sort()

string       = "', '".join(systemsName)
log(f"Found Systems: '{string}'")


types  = classified_complete[typeColumn].to_list()
counts      = Counter(types)

log("Logging the Source Types of the data.")
for key in counts.keys():
    log(f"Found {counts[key]} synonyms having the type '{key}' in the data (~{int(counts[key] / len(systemsName))} per system).")


classes = classified_complete[classColumn].tolist()
counts      = Counter(classes)

log("Logging the Semantic Classes of the data.")
for key in counts.keys():
    log(f"Found {counts[key]} synonyms having the class '{key}' in the data (~{int(counts[key] / len(systemsName))} per system).")


ontologies  = [str(s).split(":", 1)[0] for s in classified_complete[hpoidColumn].to_list()]
counts      = Counter(ontologies)

log("Logging the Ontologies of the data.")
for key in counts.keys():
    if (counts[key] > 1000):
        log(f"Found {counts[key]} entries in HPO for ontology '{key}'.")


classified   = classified_complete.copy()

log(f"Classified Synonyms: {len(classified.index)} " \
    f"(~{int(len(classified.index) / len(systemsName))} per system)")

systems                  = list(set(classified[systemColumn].tolist()))
classificationClassTypes = list(set(classified[typeColumn]))

colors      = plt.cm.tab10(range(len(systems) + 1))
metrics     = [f1ScoreLabel, recallLabel, precisionLabel]
x           = np.arange(len(systems)) * 0.2
bar_width   = 0.2
loc_legend  = "lower center"






result = {}

for system in systems:
    systemData = classified[classified[systemColumn] == system]

    systemResults = {}

    if systemData is not None and len(systemData.index) > 0:
        for classificationClassType in classificationClassTypes:
            systemResults[classificationClassType] = getMetrics(
                systemData, 
                typeColumn, 
                answerColumn, 
                classificationClassType
            )

    result[system] = systemResults

fig, axes   = plt.subplots(
    nrows   = len(metrics),
    ncols   = len(classificationClassTypes),
    figsize = (3 * len(metrics), 3 * len(classificationClassTypes)),
    sharey  = True
)

handles     = [
    plt.Rectangle((0, 0), 1, 1, color = colors[i])
    for i in range(0, len(systems))
]

fig.legend(handles, systems, loc = loc_legend, ncol = 1, 
    frameon = False)

for i, c in enumerate(classificationClassTypes):
    for j, m in enumerate(metrics):
        ax      = axes[j, i]
        values  = [result[system][c][m] for system in systems]

        for k, system in enumerate(systems):
            bars = ax.bar(x[k], values[k], bar_width, color = colors[k])

            for bar in bars:
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.01,
                    f"{h:.2f}",
                    ha          = "center",
                    va          = "bottom",
                    fontsize    = 7
                )

        ax.set_ylim(0, 1)
        ax.set_xticks(x)
        ax.set_xticklabels([""] * len(systems))

        if i == 0:
            ax.set_ylabel(m.capitalize())
        if j == 0:
            ax.set_title(str(c).capitalize())

        ax.grid(axis = "y")

fig.suptitle("Per-Class and Per-Metric Comparison", fontsize = 14)
plt.tight_layout(rect = [0, 0.05, 1, 1])
plt.savefig(outputFileClassificationTypeRecallPrecisionF1, dpi = 300, 
    bbox_inches = "tight")








classified = classified_complete.copy().drop([answerColumn, systemColumn], axis=1).drop_duplicates().reset_index(drop = True)

classes = Counter(classified[typeColumn])

# Prepare data for plotting
labels = list(classes.keys())
values = list(classes.values())

# Create the bar plot
plt.figure()
bars = plt.bar(labels, values, color = colors)

# Add counts on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        str(height),
        ha='center',
        va='bottom'
    )

plt.xlabel("Source Type")
plt.ylabel("Count")
plt.title("Count of Source Type")
plt.grid(axis = "y")
plt.show()
plt.savefig(outputFileClassificationTypeGoldCounts, dpi = 300, 
    bbox_inches = "tight")







# Count occurrences per system and classification
classified = classified_complete[typeColumn].isin([expertSynonymType, laypersonSynonymType]).copy().reset_index(drop = True)
counts = classified_complete.groupby([answerColumn, systemColumn]).size().unstack(fill_value = 0)

# Create plot
ax = counts.plot(kind = "bar", figsize = (3 * len(systems), 4), width = 0.8)

# Add value labels above bars
for container in ax.containers:
    ax.bar_label(container, padding = 3)

plt.xlabel("Classified Source Type")
plt.ylabel("Count")
plt.title("Count of Classified Source Type")
plt.xticks(rotation = 0)
plt.grid(axis = "y")
plt.show()
plt.savefig(outputFileClassificationTypeAnswerCounts, dpi = 300, 
    bbox_inches = "tight")






fig, axes   = plt.subplots(
    nrows   = 1,
    ncols   = len(metrics),
    figsize = (3 * len(metrics), 1),
    sharey  = True
)

handles = [
    plt.Rectangle((0, 0), 1, 1, color = colors[i])
    for i in range(0, len(systems))
]

fig.legend(handles, systems, loc = loc_legend, ncol = 1, 
    frameon = False)

for j, metric in enumerate(metrics):
    ax      = axes[j]
    values  = [result[system][expertSynonymType][metric] for system in systems]

    for k, system in enumerate(systems):
        bars = ax.bar(x[k], values[k], bar_width, color = colors[k])

        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h - 0.15,
                f"{h:.2f}",
                ha          = "center",
                va          = "bottom",
                fontsize    = 7
            )

    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels([""] * len(systems))
    ax.set_title(metric.capitalize())
    ax.grid(axis = "y")

fig.suptitle("Expert Synonym Type Comparison", fontsize = 14)
plt.tight_layout(rect = [0, 0.05, 1, 1])
plt.savefig(outputFileClassificationEvaluationExact, dpi = 300, 
    bbox_inches = "tight")



classified = classified_complete[classified_complete[hpoidColumn].str.startswith("HP:", na = False)].reset_index(drop = True).copy()


if len(classified.index) > 0:
    result = {}

    for system in systems:
        systemData = classified[classified[systemColumn] == system]

        systemResults = {}

        if systemData is not None and len(systemData.index) > 0:
            for classificationClass in classificationClasses:
                systemResults[classificationClass] = getMetrics(
                    systemData, 
                    typeColumn, 
                    answerColumn, 
                    classificationClass
                )

        result[system] = systemResults

    fig, axes   = plt.subplots(
        nrows   = len(classificationClasses),
        ncols   = len(metrics),
        figsize = (3 * len(metrics), 2 * len(classificationClasses)),
        sharey  = True
    )

    handles = [
        plt.Rectangle((0, 0), 1, 1, color = colors[i])
        for i in range(0, len(systems))
    ]

    fig.legend(handles, systems, loc = loc_legend, ncol = 1, 
        frameon = False)

    for i in [0, 1]:
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            values = [result[system][expertSynonymType][metric] for system in systems]

            for k, system in enumerate(systems):
                bars = ax.bar(x[k], values[k], bar_width, color = colors[k])

                for bar in bars:
                    h = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        h - 0.15,              # small vertical offset
                        f"{h:.2f}",
                        ha          = "center",
                        va          = "bottom",
                        fontsize    = 7
                    )

            ax.set_ylim(0, 1)
            ax.set_xticks(x)
            ax.set_xticklabels([""] * len(systems))
            ax.set_title(metric.capitalize())
            ax.grid(axis = "y")

    fig.suptitle("Exact Synonym Class Comparison Across Systems for HPO Only Concepts", fontsize = 14)
    plt.tight_layout(rect = [0, 0.05, 1, 1])
    plt.savefig(outputFileClassificationTypeEvaluationExactHPO, dpi = 300, bbox_inches = "tight")
else:
    log("No HPO Data available.")



classified = classified_complete[classified_complete[hpoidColumn].str.startswith("UBERON:", na = False)].reset_index(drop = True).copy()

if len(classified.index) > 0:
    result = {}

    for system in systems:
        systemData = classified[classified[systemColumn] == system]

        systemResults = {}

        if systemData is not None and len(systemData.index) > 0:
            for classificationClass in classificationClasses:
                systemResults[classificationClassType] = getMetrics(
                    systemData, 
                    typeColumn, 
                    answerColumn, 
                    classificationClassType
                )

        result[system] = systemResults

    fig, axes   = plt.subplots(
        nrows   = len(classificationClasses),
        ncols   = len(metrics),
        figsize = (3 * len(metrics), 2 * len(classificationClasses)),
        sharey  = True
    )

    handles = [
        plt.Rectangle((0, 0), 1, 1, color = colors[i])
        for i in range(0, len(systems))
    ]

    fig.legend(handles, systems, loc = loc_legend, ncol = 1, 
        frameon = False)

    for i in [0, 1]:
        for j, metric in enumerate(metrics):
            ax      = axes[i, j]
            if classificationClasses[i] in result[system].keys():
                values  = [result[system][classificationClasses[i]][metric] for system in systems]

                for k, system in enumerate(systems):
                    bars = ax.bar(x[k], values[k], bar_width, color = colors[k])

                    for bar in bars:
                        h = bar.get_height()
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            h - 0.15,              # small vertical offset
                            f"{h:.2f}",
                            ha          = "center",
                            va          = "bottom",
                            fontsize    = 7
                        )

            ax.set_ylim(0, 1)
            ax.set_xticks(x)
            ax.set_xticklabels([""] * len(systems))
            ax.set_title(metric.capitalize())
            ax.grid(axis = "y")

    fig.suptitle("Exact Synonym Class Comparison Across Systems for UBERON Only Concepts", fontsize = 14)
    plt.tight_layout(rect = [0, 0.05, 1, 1])
    plt.savefig(outputFileClassificationTypeEvaluationExactUBERON, dpi = 300, bbox_inches = "tight")
else:
    log("No UBERON Data available.")


classified = classified_complete[classified_complete[hpoidColumn].str.startswith("GO:", na = False)].reset_index(drop = True).copy()

if len(classified.index) > 0:
    result = {}

    for system in systems:
        systemData = classified[classified[systemColumn] == system]

        systemResults = {}

        if systemData is not None and len(systemData.index) > 0:
            for classificationClass in classificationClasses:
                systemResults[classificationClassType] = getMetrics(
                    systemData, 
                    typeColumn, 
                    answerColumn, 
                    classificationClassType
                )

        result[system] = systemResults

    fig, axes   = plt.subplots(
        nrows   = len(classificationClasses),
        ncols   = len(metrics),
        figsize = (3 * len(metrics), 2 * len(classificationClasses)),
        sharey  = True
    )

    handles = [
        plt.Rectangle((0, 0), 1, 1, color = colors[i])
        for i in range(0, len(systems))
    ]

    fig.legend(handles, systems, loc = loc_legend, ncol = 1, 
        frameon = False)

    for i in [0, 1]:
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            if classificationClasses[i] in result[system].keys():
                values = [result[system][classificationClasses[i]][metric] for system in systems]

                for k, system in enumerate(systems):
                    bars = ax.bar(x[k], values[k], bar_width, color = colors[k])

                    for bar in bars:
                        h = bar.get_height()
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            h - 0.15,              # small vertical offset
                            f"{h:.2f}",
                            ha          = "center",
                            va          = "bottom",
                            fontsize    = 7
                        )

            ax.set_ylim(0, 1)
            ax.set_xticks(x)
            ax.set_xticklabels([""] * len(systems))
            ax.set_title(metric.capitalize())
            ax.grid(axis = "y")

    fig.suptitle("Exact Synonym Class Comparison Across Systems for GO Only Concepts", fontsize = 14)
    plt.tight_layout(rect = [0, 0.05, 1, 1])
    plt.savefig(outputFileClassificationTypeEvaluationExactGO, dpi = 300, bbox_inches = "tight")
else:
    log("No GO Data available.")




classified              = classified_complete[
    classified_complete[hpoidColumn].str.startswith("CHEBI:", na = False)]
classified              = classified.reset_index(drop = True).copy()

if len(classified.index) > 0:
    result = {}

    for system in systems:
        systemData = classified[classified[systemColumn] == system]

        systemResults = {}

        if systemData is not None and len(systemData.index) > 0:
            for classificationClass in classificationClasses:
                systemResults[classificationClassType] = getMetrics(
                    systemData, 
                    typeColumn, 
                    answerColumn, 
                    classificationClassType
                )

        result[system] = systemResults

    fig, axes   = plt.subplots(
        nrows   = len(classificationClasses),
        ncols   = len(metrics),
        figsize = (3 * len(metrics), 2 * len(classificationClasses)),
        sharey  = True
    )

    handles = [
        plt.Rectangle((0, 0), 1, 1, color = colors[i])
        for i in range(0, len(systems))
    ]

    fig.legend(handles, systems, loc = loc_legend, ncol = 1, 
        frameon = False)

    for i in [0, 1]:
        for j, metric in enumerate(metrics):
            ax = axes[i,j]
            if classificationClasses[i] in result[system].keys():
                values = [result[system][classificationClasses[i]][metric] for system in systems]

                for k, system in enumerate(systems):
                    bars = ax.bar(x[k], values[k], bar_width, color = colors[k])

                    for bar in bars:
                        h = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width() / 2, h - 0.15,f"{h:.2f}",
                            ha = "center", va = "bottom", fontsize = 7)

            ax.set_ylim(0, 1)
            ax.set_xticks(x)
            ax.set_xticklabels([""] * len(systems))
            ax.set_title(metric.capitalize())
            ax.grid(axis = "y")

    fig.suptitle("Exact Synonym Class Comparison Across Systems for CHEBI Only Concepts", fontsize = 14)
    plt.tight_layout(rect = [0, 0.05, 1, 1])
    plt.savefig(outputFileClassificationTypeEvaluationExactCHEBI, dpi = 300, bbox_inches = "tight")
else:
    log("No CHEBI Data available.")






classified              = classified_complete
result = {}

for index, row in classified.iterrows():
    if row[typeColumn] is not None:
        string = str(row[hpoidColumn]) + " " + str(row[contentColumn])

        if string not in result.keys():
            result[string] = {
                "correct" : str(row[typeColumn])
            }
        
        if row[answerColumn] in result[string].keys():
            result[string][row[answerColumn]] = result[string][row[answerColumn]] + 1
        else:
            result[string][row[answerColumn]] = 1

resultCount = {
    "True Positive Layperson" : 0,
    "True Positive Expert" : 0,
    "True Negative Layperson" : 0,
    "True Negative Expert" : 0,
    "False Positive Layperson" : 0,
    "False Positive Expert" : 0,
    "False Negative Layperson" : 0,
    "False Negative Expert" : 0,
    "correct" : 0,
    "correctVoteCount" : [0 for _ in range(0, len(systems))],
    "voteCount" : [0 for _ in range(0, len(systems))]
}
for key in result.keys():
    votes = 0
    finalAnswer = ""

    for answer in result[key].keys():
        if isinstance(result[key][answer], int) and result[key][answer] >= votes:
            votes = result[key][answer]
            finalAnswer = answer
            if result[key][answer] == votes:
                log(f"Uncertain-Problem with {key}.")

    if finalAnswer == result[key]["correct"]:
        resultCount["correct"] = resultCount["correct"] + 1
        resultCount["correctVoteCount"][votes] = resultCount["correctVoteCount"][votes] + 1
        if finalAnswer == laypersonSynonymType:
            resultCount["True Positive Layperson"] = resultCount["True Positive Layperson"] + 1
            resultCount["True Negative Expert"] = resultCount["True Negative Expert"] + 1
        else:
            resultCount["True Positive Expert"] = resultCount["True Positive Expert"] + 1
            resultCount["True Negative Layperson"] = resultCount["True Negative Layperson"] + 1
    else:
        if finalAnswer == laypersonSynonymType:
            resultCount["False Positive Layperson"] = resultCount["False Positive Layperson"] + 1
            resultCount["False Negative Expert"] = resultCount["False Negative Expert"] + 1
        else:
            resultCount["False Negative Layperson"] = resultCount["False Negative Layperson"] + 1
            resultCount["False Positive Expert"] = resultCount["False Positive Expert"] + 1

    resultCount["voteCount"][votes] = resultCount["voteCount"][votes] + 1

log(str(resultCount))
    

# outputFileClassificationTypeEvaluationMajority






minutes         = int((time.time() - start_time) // 60)

# Print a formatted header indicating the end of this processing stage
printHeader(f"Data Evaluated [Minutes: {minutes}]")