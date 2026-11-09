"""
The preparation run and the first stage of syngen: transforms the HPO, draws
the development and evaluation set and the few-shot examples (preparation),
and generates synonyms of the concepts of an experiment with the LLMs of
llm.model_id (main()). Every concept is prompted syngen.llm.rounds times, each
round with its own seed; a synonym records in how many rounds it was
generated. The model package is imported by the step that needs it, so the
preparation run does not load vLLM.

    python3 ./src/SynGenLLM.py                  preparation
    python3 ./src/SynGenLLM.py <index>          experiment <index>, all models
    python3 ./src/SynGenLLM.py <index> <model>  only this model
"""

import os
import sys

import pandas as pd

sys.dont_write_bytecode = True

from SynGenUtils import *
from config      import *

from logger      import Logger

l = Logger(config = configFile)

# ==============================================================================
# Prompts: the model loads the template (llm.prompt_files) and fills in the
# values given here.
# ==============================================================================

# Index of the generation template in llm.prompt_files.
generationPromptIndex = 0

def examples(fewShot : bool = fewShot) -> str:
    """
    The value of {examples}: two line breaks and the few-shot examples
    (hpo.fewshot.json), each with its input and its answer in the format of
    the prompt; nothing without few-shot.
    """
    if not fewShot:
        return ""

    ret = "\n\nExamples"
    for index, example in enumerate(readFewShotExamples()):
        answer = json.dumps({synonymsKey : example["synonyms"]}, ensure_ascii = False, indent = "\t")
        ret += f"""

Example {index + 1}

Input:
* Label: {quote(example["label"])}{promptDetails(example)}

Output:
{answer}"""
    return ret

def promptValues(information : dict, examplesValue : str) -> dict:
    """
    The values of the placeholders of the generation prompt (see
    prompts/README.md).
    """
    return {
        "hpo_label"    : quote(information["label"]),
        "hpo_details"  : promptDetails(information),
        "examples"     : examplesValue,
        "max_synonyms" : syngenConfig.llm.max_synonyms
    }

# ==============================================================================
# Generation of the synonyms with the LLM -> the raw answers and the LLM
# stage file.
# ==============================================================================

def countRounds(found : pd.DataFrame) -> pd.DataFrame:
    """
    The synonyms of the rounds (one row per concept, round and synonym, with
    its lexicalKey() in the column "key") as one row per concept and synonym:
    in its most frequent spelling (on a tie the first generated), with the
    number of rounds that generated it and the first of them.
    """
    keys = [idColumn, "key"]
    spellings = found.groupby(keys + [synonymColumn], sort = False).size().reset_index(name = "count")
    spellings = spellings.sort_values("count", ascending = False, kind = "stable") \
        .drop_duplicates(subset = keys)
    counted = found.groupby(keys, sort = False).agg(**{
        roundsColumn     : ("round", "nunique"),
        firstRoundColumn : ("round", "min")
    }).reset_index()
    counted = counted.merge(spellings[keys + [synonymColumn]], on = keys, how = "left")
    counted[sourceColumn] = llmStage
    return synonymFrame(counted.drop(columns = "key").to_dict("records"))

def generateSynonyms(modelID : str) -> pd.DataFrame:
    """
    Generates the synonyms of the concepts of the experiment with one model,
    syngen.llm.rounds times, and returns one row per concept and synonym
    (by lexicalKey()) with the number of rounds that generated it. Writes the
    answers of every round to the raw file.
    """
    from model import Model

    modelIndex = configurationFile["llm"]["model_id"].index(modelID)
    modelName  = modelNameOf(modelID)

    l.printHeader(f"Generating the Synonyms ({modelName})")

    information = conceptInformation(readTask())
    concepts    = [hpoID for hpoID in taskConcepts() if hpoID in information]
    if len(concepts) == 0:
        raise ValueError(f"{quote(os.path.basename(inputFileTask))} contains no concept to " \
            "generate synonyms for.")

    examplesValue = examples()
    values = [promptValues(information[hpoID], examplesValue) for hpoID in concepts]

    settings = [name for name, value in experiment.model_dump().items() if value is True]
    l.log(f"Experiment {experimentIndex}: {applyFormat(settings)}, stages " \
        f"{applyFormat(experimentStages)}.")
    l.log(f"Generating synonyms for {len(concepts)} concepts in {rounds} rounds.")

    seed  = configurationFile["llm"]["seed"]
    # The answers of all rounds, and one row per synonym of an answer.
    answers, found = [], []
    with l.step(f"Setting up the LLM ({modelName})"):
        model = Model(configFile, modelIndex)
    with model:
        for r in range(1, rounds + 1):
            with l.step(f"Round {r}/{rounds} (seed {seed + r - 1})"):
                # The model stays loaded; each round has its own answer store
                # and prompt log, as a store refuses answers of another seed.
                model.setSeed(seed + r - 1)
                model.setPostfix(f"{generationStr}_round{r}")
                model.reset()

                model.addPromptsFromFile(generationPromptIndex, values)
                if r == 1:
                    l.log(f"Prompt for the first concept ({quote(concepts[0])}):\n" \
                        f"{model.getPrompt(0, 0)}", cmdline = False)
                model.generate()

                invalid, count = 0, 0
                for index, hpoID in enumerate(concepts):
                    terms = parseTerms(model.toJSON(index), synonymsKey)
                    answers.append({
                        idColumn    : hpoID,
                        "round"     : r,
                        "answer"    : model.getAnswer(index, 0),
                        "valid"     : terms is not None,
                        synonymsKey : json.dumps(terms or [], ensure_ascii = False)
                    })
                    invalid += terms is None
                    # A synonym listed twice in an answer counts once.
                    seen = set()
                    for term in terms or []:
                        key = lexicalKey(term)
                        if key != "" and key not in seen:
                            seen.add(key)
                            found.append({idColumn : hpoID, "round" : r, synonymColumn : term, "key" : key})
                    count += len(seen)
                if invalid > 0:
                    l.warning(f"Round {r}: {invalid} of {len(concepts)} answers contain no list " \
                        "of synonyms (e.g. cut off at llm.max_tokens or longer than llm.max_model_len).")
                l.log(f"Round {r}: {count} synonyms.")
                model.logPrompts()

    writeHugeCSV(pd.DataFrame(answers), outputFileLLMRaw(modelName))

    found = pd.DataFrame(found, columns = [idColumn, "round", synonymColumn, "key"])
    synonyms = countRounds(found)
    synonyms[modelColumn] = modelName
    synonyms = markAsLabel(synonyms, {hpoID : information[hpoID]["label"] for hpoID in concepts})

    l.log(f"{len(synonyms.index)} distinct synonyms of {synonyms[idColumn].nunique()} of " \
        f"{len(concepts)} concepts; {synonyms[asLabelColumn].sum()} equal their label.")
    for count, number in synonyms[roundsColumn].value_counts().sort_index().items():
        l.log(f"Generated in {count} of {rounds} rounds: {number} synonyms.")

    return synonyms

# ==============================================================================
# The steps of this run.
# ==============================================================================

def main() -> None:
    """
    Runs the steps: the preparation, and without an experiment index nothing
    else; otherwise the generation of the experiment with every model.
    """
    startRun(experimentRequired = False)

    prepare()
    if experimentIndex is None:
        return

    runForModels("Generating the synonyms", lambda modelID: writeHugeCSV(
        generateSynonyms(modelID), outputFileStage(llmStage, modelNameOf(modelID))))


# vLLM starts its workers with "spawn", which imports this script again: the
# steps must only run in the main process.
if __name__ == "__main__":
    main()
