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
import shlex
import logging

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

def promptDetails(information : dict, definition : bool = addDefinition,
        comment : bool = addComment, structure : bool = addStructure) -> str:
    """
    The value of {hpo_details}: one line per information the experiment adds
    (definition, comment, and parents and children for structure), each
    starting with a line break; empty without them.
    """
    details = []
    if definition:
        details.append(f"* Definition: {quote(information['definition'])}")
    if comment:
        details.append(f"* Comment: {quote(information['comment'])}")
    if structure:
        details.append(f"* Parents: {applyFormat(information['parents'])}")
        details.append(f"* Children: {applyFormat(information['children'])}")
    return "".join(f"\n{line}" for line in details)

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

def useSeed(model, seed : int) -> None:
    """
    Makes the loaded model sample with another seed: its configuration (which
    the answer store records, so answers of another seed are never reused)
    and its sampling parameters, without loading the model again.
    """
    from vllm import SamplingParams

    model.config = model.config.model_copy(update = {"seed" : seed})
    model.sampling_params = SamplingParams(
        temperature = model.config.temperature,
        top_p       = model.config.top_p,
        max_tokens  = model.config.max_tokens,
        seed        = seed
    )

def generateSynonyms(modelID : str) -> pd.DataFrame:
    """
    Generates the synonyms of the concepts of the experiment with one model,
    syngen.llm.rounds times, and returns one row per concept and synonym
    (by lexicalKey()) with the number of rounds that generated it. Writes the
    answers of every round to the raw file.
    """
    from model import Model

    logging.getLogger("vllm").setLevel(logging.ERROR)

    if modelID not in configurationFile["llm"]["model_id"]:
        raise ValueError(f"{quote(modelID)} is not in llm.model_id.")
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

    seed = configurationFile["llm"]["seed"]
    answers = []
    l.log(f"Setting up the LLM ({modelName})...")
    with Model(configFile, modelIndex) as model:
        l.log(f"Setting up the LLM ({modelName}) completed.")

        for r in range(1, rounds + 1):
            with l.step(f"Round {r}/{rounds} (seed {seed + r - 1})"):
                # Each round has its own answer store and prompt log, as the
                # store refuses answers of another seed.
                useSeed(model, seed + r - 1)
                model.setPostfix(f"{generationStr}_round{r}")
                model.reset()

                model.addPromptsFromFile(generationPromptIndex, values)
                if r == 1:
                    l.log(f"Prompt for the first concept ({quote(concepts[0])}):\n" \
                        f"{model.getPrompt(0, 0)}", cmdline = False)
                model.generate()

                parsed = [parseTerms(model.toJSON(index), synonymsKey) for index in range(len(values))]
                for index, (hpoID, terms) in enumerate(zip(concepts, parsed)):
                    answers.append({
                        idColumn      : hpoID,
                        "round"       : r,
                        "answer"      : model.getAnswer(index, 0),
                        "valid"       : terms is not None,
                        synonymsKey   : json.dumps(terms or [], ensure_ascii = False)
                    })
                invalid = sum(terms is None for terms in parsed)
                if invalid > 0:
                    l.warning(f"Round {r}: {invalid} of {len(parsed)} answers contain no list " \
                        "of synonyms (e.g. longer than llm.max_model_len).")
                l.log(f"Round {r}: {sum(len(t or []) for t in parsed)} synonyms.")
                model.logPrompts()

    raw = pd.DataFrame(answers)
    writeHugeCSV(raw, outputFileLLMRaw(modelName))

    # One row per concept and synonym; its most frequent spelling, and the
    # number of rounds that generated it.
    rows = []
    for (hpoID, r), group in raw.groupby([idColumn, "round"], sort = False):
        for synonym in dict.fromkeys(json.loads(group[synonymsKey].iloc[0])):
            rows.append({idColumn : hpoID, "round" : r, synonymColumn : synonym,
                "key" : lexicalKey(synonym)})
    found = pd.DataFrame(rows, columns = [idColumn, "round", synonymColumn, "key"])
    found = found[found["key"] != ""]

    grouped = found.groupby([idColumn, "key"], sort = False)
    synonyms = synonymFrame([
        {
            idColumn         : hpoID,
            synonymColumn    : group[synonymColumn].value_counts().index[0],
            sourceColumn     : llmStage,
            modelColumn      : modelName,
            roundsColumn     : group["round"].nunique(),
            firstRoundColumn : group["round"].min()
        }
        for (hpoID, _), group in grouped
    ])
    labels = {hpoID : information[hpoID]["label"] for hpoID in concepts}
    synonyms = markAsLabel(synonyms, labels)

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
    # The log file is named after the start of the run only, so the command
    # line is its first line.
    l.log(f"Command: python3 {shlex.join(sys.argv)}")

    prepare()
    if experimentIndex is None:
        return

    # A failing model does not stop the others; each is unloaded from the GPU
    # before the next one.
    failed = []
    for index, modelID in enumerate(modelIDs):
        l.log(f"Model {index + 1}/{len(modelIDs)}: {quote(modelID)}.")
        modelName = modelNameOf(modelID)
        try:
            with l.step(f"Generating the synonyms ({modelName})"):
                synonyms = generateSynonyms(modelID)
            writeHugeCSV(synonyms, outputFileStage(llmStage, modelName))
        except (Exception, SystemExit):
            # The step's rows give the reason; the traceback goes to the log
            # file only.
            l.exception(f"Generating with {quote(modelID)} failed.", cmdline = False)
            failed.append(modelID)

    if len(failed) > 0:
        l.error(f"{len(failed)} of {len(modelIDs)} model(s) failed: {applyFormat(failed)}.")
        sys.exit(1)


# vLLM starts its workers with "spawn", which imports this script again: the
# steps must only run in the main process.
if __name__ == "__main__":
    main()
