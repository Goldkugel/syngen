"""
The configuration of syngen: reads config/config.yaml, picks the experiment
and the models of a run from the command line, and defines the column names
and the file names of every step. Imported by every script with
`from config import *`.

    python3 ./src/config.py   prints the number of experiments
"""

import os
import sys
import yaml

from SynGenConfig import ExperimentConfig, SyngenConfig
from SynGenConfig import stages, llmStage, pubmedStage, conceptnetStage

sys.dont_write_bytecode = True

# =============================================================================
# Configuration File
# =============================================================================

# The settings of syngen ("syngen" section) and of the packages it uses
# (adapter, embedder, llm, logger, normalizer).
configFile = "./config/config.yaml"

with open(configFile, "r", encoding = "utf-8") as configHandle:
    configurationFile = yaml.safe_load(configHandle)

syngenConfig = SyngenConfig.model_validate(configurationFile["syngen"])

# =============================================================================
# Experiment and models of a run, given on the command line:
#   (nothing)           the preparation run (SynGenLLM.py only);
#   <index>             the experiment syngen.experiments[<index>], from 0,
#                       with every model of llm.model_id;
#   <index> <model ID>  only this model.
# Options (e.g. --fetch) are checked by the scripts (startRun() in
# SynGenUtils.py), which know the options they take.
# =============================================================================

arguments = [argument for argument in sys.argv[1:] if not argument.startswith("--")]
options   = [argument for argument in sys.argv[1:] if argument.startswith("--")]

if len(arguments) > 2:
    sys.exit(f"Too many arguments: {' '.join(arguments)}; give the experiment index and " \
        "at most one model ID.")

experimentIndex = None
if len(arguments) > 0:
    experimentCount = len(syngenConfig.experiments)
    if not arguments[0].isdigit() or int(arguments[0]) >= experimentCount:
        sys.exit(f"The experiment '{arguments[0]}' does not exist: give its index in " \
            f"syngen.experiments, 0 to {experimentCount - 1}.")
    experimentIndex = int(arguments[0])

# The preparation run generates nothing, so it has no experiment.
experiment = (ExperimentConfig() if experimentIndex is None
    else syngenConfig.experiments[experimentIndex])

reduceToTestIDs         = experiment.test
fewShot                 = experiment.few_shot
addDefinition           = experiment.definition
addComment              = experiment.comment
addStructure            = experiment.structure
experimentStages        = experiment.stages

rounds                  = syngenConfig.llm.rounds

testStr                 = "Test" if reduceToTestIDs else "NoTest"
configStr               = "Def" if addDefinition else "NoDef"
configStr              += "_Com" if addComment else "_NoCom"
configStr              += "_Str" if addStructure else "_NoStr"
fewShotStr              = "FS" if fewShot else "NoFS"
roundsStr               = f"R{rounds}"

# Everything the LLM's synonyms depend on, e.g. Test_Def_Com_Str_FS_R5.
generationStr           = f"{testStr}_{configStr}_{fewShotStr}_{roundsStr}"

def stagesStr(stage : str) -> str:
    """
    The stages of the experiment up to and including `stage`, by their
    first letters, e.g. "LP" for the LLM and PubMed: the file of a stage
    holds the synonyms of all of them.
    """
    upTo = experimentStages[:experimentStages.index(stage) + 1]
    return "".join(s[0].upper() for s in upTo)

def previousStage(stage : str) -> str | None:
    """The stage of the experiment before `stage`, None for the first."""
    index = experimentStages.index(stage)
    return experimentStages[index - 1] if index > 0 else None

# =============================================================================
# Model Configuration
# =============================================================================

# None in the preparation run (empty), the one given after the experiment, or
# all of llm.model_id.
modelIDs = []
if experimentIndex is not None:
    modelIDs = arguments[1:2] or list(configurationFile["llm"]["model_id"])
    unknown  = [modelID for modelID in modelIDs if modelID not in configurationFile["llm"]["model_id"]]
    if len(unknown) > 0:
        sys.exit(f"The model '{unknown[0]}' is not in llm.model_id.")

def modelNameOf(modelID : str) -> str:
    """
    The name of a model in file names and the model column: its ID without
    the organisation, e.g. "Qwen2.5-72B-Instruct-GPTQ-Int4".
    """
    return modelID[modelID.index("/") + 1:] if "/" in modelID else modelID

# The embedding model of the similarities: the first of embedder.model_id,
# named by syngen.embedding_model_names or its ID without the organisation.
embeddingModelID   = configurationFile["embedder"]["model_id"][0]
embeddingModelName = syngenConfig.embedding_model_names.get(embeddingModelID,
    modelNameOf(embeddingModelID))

# =============================================================================
# Formatting of the prompts and the log
# =============================================================================

quotationCharacter = "\""

# =============================================================================
# Columns of the synonym files, after the concept ID (the transformed HPO's
# id column, see SynGenUtils.py)
# =============================================================================

# The generated synonym, and the stage (llm, pubmed, conceptnet) and model
# it comes from.
synonymColumn               = "synonym"
sourceColumn                = "source"
modelColumn                 = "model"
# LLM: in how many rounds the synonym was generated, and in which first.
roundsColumn                = "rounds"
firstRoundColumn            = "first_round"
# PubMed and ConceptNet: the term the synonym was found from, where (PMIDs or
# the ConceptNet replacements), how often, and its similarity to the label.
seedColumn                  = "seed"
evidenceColumn              = "evidence"
frequencyColumn             = "frequency"
similarityColumn            = f"similarity_{embeddingModelName}"
# Whether the synonym equals the label of its concept (normalised); such
# synonyms are reported as "as_label" and not scored.
asLabelColumn               = "as_label"
# SynGenEmbed.py: the normalised synonym (its sorted tokens) and a key equal
# for synonyms with the same embedding.
normalizedColumn            = "normalized"
embeddingKeyColumn          = "embedding_key"
# The highest similarity of a generated synonym to a gold synonym of its
# concept, and of a gold synonym to the synonyms of each stage.
goldSimilarityColumn        = "gold_similarity"
def stageSimilarityColumn(stage : str) -> str:
    """The highest similarity of a gold synonym to the synonyms of a stage."""
    return f"similarity_{stage}"

# The answer keys of the prompts (see prompts/).
synonymsKey                 = "synonyms"
keepKey                     = "keep"

# =============================================================================
# Folder structure
# =============================================================================

csvFileFormat  = "csv"
jsonFileFormat = "json"
plotFileFormat = "png"

dataDir = syngenConfig.data_folder

outputFolderName = "output"

def outputPath(folder : str, name : str) -> str:
    """A file in data/output/<folder>/."""
    return os.path.join(dataDir, outputFolderName, folder, name)

# The HPO file read by the adapter (adapter.hpo section).
inputFileTransformed        = os.path.join(
    configurationFile["adapter"]["hpo"]["input_folder"],
    configurationFile["adapter"]["hpo"]["input_files"][0]
)

# hp.owl transformed by the preparation run, where adapter.hpo.output_folder
# and output_file say. The development set, the evaluation set and the
# few-shot examples are named after it in the same folder.
outputFileTransformedFull = os.path.join(
    configurationFile["adapter"]["hpo"]["output_folder"],
    configurationFile["adapter"]["hpo"]["output_file"]
)
transformedStem, transformedExtension = os.path.splitext(outputFileTransformedFull)

# The development set (syngen.development_set): the concepts drawn at random,
# with their information and the labels of their parents and children.
outputFileTransformedDevelopment = f"{transformedStem}.development{transformedExtension}"

# The evaluation set: hpo.csv without the synonyms of the development
# concepts, whose labels may still be needed as parents or children.
outputFileTransformedEvaluation  = f"{transformedStem}.evaluation{transformedExtension}"

# The few-shot examples, drawn from the development set.
outputFileFewShot                = f"{transformedStem}.fewshot.{jsonFileFormat}"

inputFileTask = outputFileTransformedDevelopment if reduceToTestIDs else outputFileTransformedEvaluation

# =============================================================================
# Files of the stages: the synonyms of a stage and of the stages before it,
# one file per model, e.g.
# data/output/pubmed/pubmed_Test_Def_Com_Str_FS_R5_LP_<model>.csv.
# =============================================================================

def outputFileStage(stage : str, modelName : str) -> str:
    """The synonyms of the stages of the experiment up to `stage`."""
    return outputPath(stage,
        f"{stage}_{generationStr}_{stagesStr(stage)}_{modelName}.{csvFileFormat}")

def outputFileLLMRaw(modelName : str) -> str:
    """The prompts and answers of every round of the LLM."""
    return outputPath(llmStage,
        f"{llmStage}_{generationStr}_raw_{modelName}.{csvFileFormat}")

# The steps of a stage that need the model package and those that need the
# normalizer run in different Python environments (see generate_llm.sh,
# generate_sources.sh and filter_llm.sh) and hand their results on in these
# files.

def outputFileLLMUnmarked(modelName : str) -> str:
    """The synonyms of the LLM before markAsLabel() (SynGenLLM.py --mark)."""
    return outputPath(llmStage,
        f"{llmStage}_{generationStr}_unmarked_{modelName}.{csvFileFormat}")

def outputFileConceptNetSpans(modelName : str) -> str:
    """The seeds, their spans and the neighbours of the spans (--spans)."""
    return outputPath(conceptnetStage,
        f"{conceptnetStage}_{generationStr}_{stagesStr(conceptnetStage)}_spans_{modelName}.{jsonFileFormat}")

def outputFileConceptNetFilter(modelName : str) -> str:
    """The neighbours the LLM kept per concept and span (--filter)."""
    return outputPath(conceptnetStage,
        f"{conceptnetStage}_{generationStr}_{stagesStr(conceptnetStage)}_filter_{modelName}.{jsonFileFormat}")

# The last stage of the experiment, whose file holds all synonyms.
finalStage = experimentStages[-1]

# SynGenEmbed.py: the synonyms of all stages with their normalised form and
# similarities, and the gold synonyms with their best similarity per stage.
outputFolderNameEmbedded = "embedded"

def outputFileEmbedded(modelName : str) -> str:
    """The generated synonyms, normalised and embedded."""
    return outputPath(outputFolderNameEmbedded,
        f"embed_{generationStr}_{stagesStr(finalStage)}_{modelName}.{csvFileFormat}")

def outputFileEmbeddedGold(modelName : str) -> str:
    """The gold synonyms of the concepts, normalised and embedded."""
    return outputPath(outputFolderNameEmbedded,
        f"embed_{generationStr}_{stagesStr(finalStage)}_{modelName}_gold.{csvFileFormat}")

# SynGenEval.py.
outputFolderNameEvaluation = "evaluated"

def outputFileEvaluation(modelName : str, name : str, fileFormat : str = plotFileFormat) -> str:
    """An evaluation output of the experiment, e.g. ..._summary.csv."""
    return outputPath(outputFolderNameEvaluation,
        f"eval_{generationStr}_{stagesStr(finalStage)}_{modelName}_{name}.{fileFormat}")


# The number of experiments, for the shell scripts (finalize.sh).
if __name__ == "__main__":
    print(len(syngenConfig.experiments))
