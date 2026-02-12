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

printHeader("Evaluating the Results of Synonym Classification")
start_time = time.time()

# Only proceed if formatted input data exists
exitIfFileNotExist(inputFileClassificationEvaluation)

classified_complete = readCSV(inputFileClassificationEvaluation)

ontologies = [str(s).split(":", 1)[0] for s in classified_complete[hpoidColumn].to_list()]

counts = Counter(ontologies)

for key in counts.keys():
    if (counts[key] > 1000):
        log(f"Found {counts[key]} entries in HPO for ontology '{key}'.")


classified = classified_complete

systemsName = list(set(classified[systemColumn].tolist()))

string = "', '".join(systemsName)
log(f"Found Systems: '{string}'")
log(f"Classified Synonyms: {len(classified.index)} (~{int(len(classified.index) / len(systemsName))} per system)")
classified   = classified[classified[systemColumn] != ""]

classified[classColumn] = classified[classColumn].str.lower()
classified[answerColumn] = classified[answerColumn].str.lower()

systems     = list(set(classified[systemColumn].tolist()))
classificationClasses = list(set(classified[classColumn]))

colors = plt.cm.tab10(range(len(systems) + 1))

result = {}

for system in systems:
    systemData = classified[classified[systemColumn] == system]

    systemResults = {}

    if systemData is not None and len(systemData.index) > 0:
        for classificationClass in classificationClasses:
            
            systemClassResults = {
                precisionLabel  : 0,
                recallLabel     : 0,
                f1ScoreLabel    : 0
            }

            if len(systemData[systemData[answerColumn] == 
                classificationClass].index) > 0:

                systemClassResults[precisionLabel] = \
                    len(systemData[
                            (systemData[classColumn] == classificationClass
                                ) & (
                            systemData[answerColumn] == classificationClass)
                        ].index
                    ) / (
                        len(systemData[systemData[answerColumn] == 
                            classificationClass].index) 
                    )
                
                systemClassResults[recallLabel] = \
                    len(systemData[
                            (systemData[classColumn] == classificationClass
                                ) & (
                            systemData[answerColumn] == classificationClass)
                        ].index
                    ) / (
                        len(systemData[systemData[classColumn] == 
                            classificationClass].index) 
                    )
                    
                systemClassResults[f1ScoreLabel] = \
                    2 * systemClassResults[precisionLabel] * \
                    systemClassResults[recallLabel] / (
                    systemClassResults[recallLabel] + 
                    systemClassResults[precisionLabel])
            else:
                systemClassResults[f1ScoreLabel]    = 0
                systemClassResults[recallLabel]     = 0
                systemClassResults[precisionLabel]  = 0

            systemResults[classificationClass] = systemClassResults

    result[system] = systemResults

metrics = [f1ScoreLabel, recallLabel, precisionLabel]
systems = list(result.keys())
classes = list(next(iter(result.values())).keys())

x = np.arange(len(systems)) * 0.1 * len(systems)
bar_width = 0.2

fig, axes = plt.subplots(
    nrows=len(metrics),
    ncols=len(classes),
    figsize=(3 * len(metrics), 2 * len(classes)),
    sharey=True
)

handles = [
    plt.Rectangle((0, 0), 1, 1, color=colors[i])
    for i in range(0, len(systems))
]

fig.legend(
    handles,
    systems,
    loc="lower center",
    ncol=len(systems),
    frameon=False
)

colors = plt.cm.tab10(range(len(systems) + 1))

for i, cls in enumerate(classes):
    for j, metric in enumerate(metrics):
        ax = axes[j, i]
        values = [result[system][cls][metric] for system in systems]

        for k, system in enumerate(systems):
            bars = ax.bar(
                x[k],
                values[k],
                bar_width,
                color=colors[k]
            )

            for bar in bars:
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.01,              # small vertical offset
                    f"{h:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7
                )

        ax.set_ylim(0, 1)
        ax.set_xticks(x)
        ax.set_xticklabels([""] * len(systems))

        if i == 0:
            ax.set_ylabel(metric.capitalize())
        if j == 0:
            ax.set_title(cls)

        ax.grid(axis="y")

fig.suptitle("Per-Class and Per-Metric Comparison Across Systems for Full HPO Concepts", fontsize=14)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(outputFileClassificationRecallPrecisionF1, dpi=300, bbox_inches="tight")



classes = [exactSynonymClass]

x = np.arange(len(systems)) * 0.2
bar_width = 0.2

fig, axes = plt.subplots(
    nrows=len(classes),
    ncols=len(metrics),
    figsize=(3 * len(metrics), 2 * len(classes)),
    sharey=True
)

handles = [
    plt.Rectangle((0, 0), 1, 1, color=colors[i])
    for i in range(0, len(systems))
]

fig.legend(
    handles,
    systems,
    loc="lower center",
    ncol=len(systems),
    frameon=False
)

colors = plt.cm.tab10(range(len(systems) + 1))

for j, metric in enumerate(metrics):
    ax = axes[j]
    values = [result[system][exactSynonymClass][metric] for system in systems]

    for k, system in enumerate(systems):
        bars = ax.bar(
            x[k],
            values[k],
            bar_width,
            color=colors[k]
        )

        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h - 0.15,              # small vertical offset
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=7
            )

    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels([""] * len(systems))

    ax.set_title(metric)

    ax.grid(axis="y")

fig.suptitle("Exact Synonym Class Comparison Across Systems for Full HPO Concepts", fontsize=14)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(outputFileClassificationEvaluationExact, dpi=300, bbox_inches="tight")



classified = classified_complete[classified_complete[hpoidColumn].str.startswith("HP:", na=False)].reset_index(drop = True).copy()

systems     = list(set(classified[systemColumn].tolist()))
classificationClasses = [exactSynonymClass]
colors = plt.cm.tab10(range(len(systems) + 1))

result = {}

for system in systems:
    systemData = classified[classified[systemColumn] == system]

    systemResults = {}

    if systemData is not None and len(systemData.index) > 0:
        for classificationClass in classificationClasses:
            
            systemClassResults = {
                precisionLabel  : 0,
                recallLabel     : 0,
                f1ScoreLabel    : 0
            }

            if len(systemData[systemData[answerColumn] == 
                classificationClass].index) > 0:

                systemClassResults[precisionLabel] = \
                    len(systemData[
                            (systemData[classColumn] == classificationClass
                                ) & (
                            systemData[answerColumn] == classificationClass)
                        ].index
                    ) / (
                        len(systemData[systemData[answerColumn] == 
                            classificationClass].index) 
                    )
                
                systemClassResults[recallLabel] = \
                    len(systemData[
                            (systemData[classColumn] == classificationClass
                                ) & (
                            systemData[answerColumn] == classificationClass)
                        ].index
                    ) / (
                        len(systemData[systemData[classColumn] == 
                            classificationClass].index) 
                    )
                    
                systemClassResults[f1ScoreLabel] = \
                    2 * systemClassResults[precisionLabel] * \
                    systemClassResults[recallLabel] / (
                    systemClassResults[recallLabel] + 
                    systemClassResults[precisionLabel])
            else:
                systemClassResults[f1ScoreLabel]    = 0
                systemClassResults[recallLabel]     = 0
                systemClassResults[precisionLabel]  = 0

            systemResults[classificationClass] = systemClassResults

    result[system] = systemResults

x = np.arange(len(systems)) * 0.2
bar_width = 0.2

fig, axes = plt.subplots(
    nrows=len(classificationClasses),
    ncols=len(metrics),
    figsize=(3 * len(metrics), 2 * len(classificationClasses)),
    sharey=True
)

handles = [
    plt.Rectangle((0, 0), 1, 1, color=colors[i])
    for i in range(0, len(systems))
]

fig.legend(
    handles,
    systems,
    loc="lower center",
    ncol=len(systems),
    frameon=False
)

colors = plt.cm.tab10(range(len(systems) + 1))

for j, metric in enumerate(metrics):
    ax = axes[j]
    values = [result[system][exactSynonymClass][metric] for system in systems]

    for k, system in enumerate(systems):
        bars = ax.bar(
            x[k],
            values[k],
            bar_width,
            color=colors[k]
        )

        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h - 0.15,              # small vertical offset
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=7
            )

    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels([""] * len(systems))

    ax.set_title(metric)

    ax.grid(axis="y")

fig.suptitle("Exact Synonym Class Comparison Across Systems for HPO Only Concepts", fontsize=14)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(outputFileClassificationEvaluationExactHPO, dpi=300, bbox_inches="tight")




classified = classified_complete[classified_complete[hpoidColumn].str.startswith("UBERON:", na=False)].reset_index(drop = True).copy()
systems     = list(set(classified[systemColumn].tolist()))
classificationClasses = [exactSynonymClass]
colors = plt.cm.tab10(range(len(systems) + 1))

result = {}

for system in systems:
    systemData = classified[classified[systemColumn] == system]

    systemResults = {}

    if systemData is not None and len(systemData.index) > 0:
        for classificationClass in classificationClasses:
            
            systemClassResults = {
                precisionLabel  : 0,
                recallLabel     : 0,
                f1ScoreLabel    : 0
            }

            if len(systemData[systemData[answerColumn] == 
                classificationClass].index) > 0:

                systemClassResults[precisionLabel] = \
                    len(systemData[
                            (systemData[classColumn] == classificationClass
                                ) & (
                            systemData[answerColumn] == classificationClass)
                        ].index
                    ) / (
                        len(systemData[systemData[answerColumn] == 
                            classificationClass].index) 
                    )
                
                systemClassResults[recallLabel] = \
                    len(systemData[
                            (systemData[classColumn] == classificationClass
                                ) & (
                            systemData[answerColumn] == classificationClass)
                        ].index
                    ) / (
                        len(systemData[systemData[classColumn] == 
                            classificationClass].index) 
                    )
                    
                systemClassResults[f1ScoreLabel] = \
                    2 * systemClassResults[precisionLabel] * \
                    systemClassResults[recallLabel] / (
                    systemClassResults[recallLabel] + 
                    systemClassResults[precisionLabel])
            else:
                systemClassResults[f1ScoreLabel]    = 0
                systemClassResults[recallLabel]     = 0
                systemClassResults[precisionLabel]  = 0

            systemResults[classificationClass] = systemClassResults

    result[system] = systemResults

x = np.arange(len(systems)) * 0.2
bar_width = 0.2

fig, axes = plt.subplots(
    nrows=len(classificationClasses),
    ncols=len(metrics),
    figsize=(3 * len(metrics), 2 * len(classificationClasses)),
    sharey=True
)

handles = [
    plt.Rectangle((0, 0), 1, 1, color=colors[i])
    for i in range(0, len(systems))
]

fig.legend(
    handles,
    systems,
    loc="lower center",
    ncol=len(systems),
    frameon=False
)

colors = plt.cm.tab10(range(len(systems) + 1))

for j, metric in enumerate(metrics):
    ax = axes[j]
    values = [result[system][exactSynonymClass][metric] for system in systems]

    for k, system in enumerate(systems):
        bars = ax.bar(
            x[k],
            values[k],
            bar_width,
            color=colors[k]
        )

        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h - 0.15,              # small vertical offset
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=7
            )

    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels([""] * len(systems))

    ax.set_title(metric)

    ax.grid(axis="y")

fig.suptitle("Exact Synonym Class Comparison Across Systems for UBERON Only Concepts", fontsize=14)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(outputFileClassificationEvaluationExactUBERON, dpi=300, bbox_inches="tight")



classified = classified_complete[classified_complete[hpoidColumn].str.startswith("GO:", na=False)].reset_index(drop = True).copy()
systems     = list(set(classified[systemColumn].tolist()))
classificationClasses = [exactSynonymClass]
colors = plt.cm.tab10(range(len(systems) + 1))

result = {}

for system in systems:
    systemData = classified[classified[systemColumn] == system]

    systemResults = {}

    if systemData is not None and len(systemData.index) > 0:
        for classificationClass in classificationClasses:
            
            systemClassResults = {
                precisionLabel  : 0,
                recallLabel     : 0,
                f1ScoreLabel    : 0
            }

            if len(systemData[systemData[answerColumn] == 
                classificationClass].index) > 0:

                systemClassResults[precisionLabel] = \
                    len(systemData[
                            (systemData[classColumn] == classificationClass
                                ) & (
                            systemData[answerColumn] == classificationClass)
                        ].index
                    ) / (
                        len(systemData[systemData[answerColumn] == 
                            classificationClass].index) 
                    )
                
                systemClassResults[recallLabel] = \
                    len(systemData[
                            (systemData[classColumn] == classificationClass
                                ) & (
                            systemData[answerColumn] == classificationClass)
                        ].index
                    ) / (
                        len(systemData[systemData[classColumn] == 
                            classificationClass].index) 
                    )
                    
                systemClassResults[f1ScoreLabel] = \
                    2 * systemClassResults[precisionLabel] * \
                    systemClassResults[recallLabel] / (
                    systemClassResults[recallLabel] + 
                    systemClassResults[precisionLabel])
            else:
                systemClassResults[f1ScoreLabel]    = 0
                systemClassResults[recallLabel]     = 0
                systemClassResults[precisionLabel]  = 0

            systemResults[classificationClass] = systemClassResults

    result[system] = systemResults

x = np.arange(len(systems)) * 0.2
bar_width = 0.2

fig, axes = plt.subplots(
    nrows=len(classificationClasses),
    ncols=len(metrics),
    figsize=(3 * len(metrics), 2 * len(classificationClasses)),
    sharey=True
)

handles = [
    plt.Rectangle((0, 0), 1, 1, color=colors[i])
    for i in range(0, len(systems))
]

fig.legend(
    handles,
    systems,
    loc="lower center",
    ncol=len(systems),
    frameon=False
)

colors = plt.cm.tab10(range(len(systems) + 1))

for j, metric in enumerate(metrics):
    ax = axes[j]
    values = [result[system][exactSynonymClass][metric] for system in systems]

    for k, system in enumerate(systems):
        bars = ax.bar(
            x[k],
            values[k],
            bar_width,
            color=colors[k]
        )

        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h - 0.15,              # small vertical offset
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=7
            )

    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels([""] * len(systems))

    ax.set_title(metric)

    ax.grid(axis="y")

fig.suptitle("Exact Synonym Class Comparison Across Systems for GO Only Concepts", fontsize=14)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(outputFileClassificationEvaluationExactGO, dpi=300, bbox_inches="tight")



classified = classified_complete[classified_complete[hpoidColumn].str.startswith("CHEBI:", na=False)].reset_index(drop = True).copy()
systems     = list(set(classified[systemColumn].tolist()))
classificationClasses = [exactSynonymClass]
colors = plt.cm.tab10(range(len(systems) + 1))

result = {}

for system in systems:
    systemData = classified[classified[systemColumn] == system]

    systemResults = {}

    if systemData is not None and len(systemData.index) > 0:
        for classificationClass in classificationClasses:
            
            systemClassResults = {
                precisionLabel  : 0,
                recallLabel     : 0,
                f1ScoreLabel    : 0
            }

            if len(systemData[systemData[answerColumn] == 
                classificationClass].index) > 0:

                systemClassResults[precisionLabel] = \
                    len(systemData[
                            (systemData[classColumn] == classificationClass
                                ) & (
                            systemData[answerColumn] == classificationClass)
                        ].index
                    ) / (
                        len(systemData[systemData[answerColumn] == 
                            classificationClass].index) 
                    )
                
                systemClassResults[recallLabel] = \
                    len(systemData[
                            (systemData[classColumn] == classificationClass
                                ) & (
                            systemData[answerColumn] == classificationClass)
                        ].index
                    ) / (
                        len(systemData[systemData[classColumn] == 
                            classificationClass].index) 
                    )
                    
                systemClassResults[f1ScoreLabel] = \
                    2 * systemClassResults[precisionLabel] * \
                    systemClassResults[recallLabel] / (
                    systemClassResults[recallLabel] + 
                    systemClassResults[precisionLabel])
            else:
                systemClassResults[f1ScoreLabel]    = 0
                systemClassResults[recallLabel]     = 0
                systemClassResults[precisionLabel]  = 0

            systemResults[classificationClass] = systemClassResults

    result[system] = systemResults

x = np.arange(len(systems)) * 0.2
bar_width = 0.2

fig, axes = plt.subplots(
    nrows=len(classificationClasses),
    ncols=len(metrics),
    figsize=(3 * len(metrics), 2 * len(classificationClasses)),
    sharey=True
)

handles = [
    plt.Rectangle((0, 0), 1, 1, color=colors[i])
    for i in range(0, len(systems))
]

fig.legend(
    handles,
    systems,
    loc="lower center",
    ncol=len(systems),
    frameon=False
)

colors = plt.cm.tab10(range(len(systems) + 1))

for j, metric in enumerate(metrics):
    ax = axes[j]
    values = [result[system][exactSynonymClass][metric] for system in systems]

    for k, system in enumerate(systems):
        bars = ax.bar(
            x[k],
            values[k],
            bar_width,
            color=colors[k]
        )

        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h - 0.15,              # small vertical offset
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=7
            )

    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels([""] * len(systems))

    ax.set_title(metric)

    ax.grid(axis="y")

fig.suptitle("Exact Synonym Class Comparison Across Systems for CHEBI Only Concepts", fontsize=14)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(outputFileClassificationEvaluationExactCHEBI, dpi=300, bbox_inches="tight")



end_time = time.time()
elapsed_seconds = end_time - start_time
minutes = int(elapsed_seconds // 60)

# Print a formatted header indicating the end of this processing stage
printHeader(f"Data Evaluated [Minutes: {minutes}]")