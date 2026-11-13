"""
Embeds and normalises the generated synonyms of an experiment (the file of
its last stage, which holds the synonyms of all stages) and the gold synonyms
of its concepts, for SynGenEval.py:

- every generated synonym gets its normalised form (normalizer package: its
  sorted tokens), its embedding key (equal for synonyms with the same
  embedding) and its highest cosine similarity to a gold synonym of its
  concept;
- every gold synonym gets its normalised form, its embedding key and its
  highest similarity to the generated synonyms of each stage.

The cosine similarity of a generated synonym to its label is computed by the
PubMed and ConceptNet stages already and filled in here for the others.

    python3 ./src/SynGenEmbed.py <index> [<model>]
"""

import hashlib
import sys

import numpy  as np
import pandas as pd

sys.dont_write_bytecode = True

from SynGenUtils import *
from config      import *

from logger      import Logger

l = Logger(config = configFile)

# Decimals of the vector components in the embedding key: synonyms whose
# normalised vectors agree in all of them (cosine similarity 1 up to
# rounding) share a key.
embeddingKeyDecimals = 4

embedOutputSettings = {
    "embedder"       : embedSettings,
    "embedding_model": embeddingModelID,
    "normalizer"     : configurationFile.get("normalizer", {}),
    "gold"           : syngenConfig.gold.model_dump(),
    "key_decimals"   : embeddingKeyDecimals,
    "stages"         : experimentStages
}

def embeddingKeys(vectors : np.ndarray) -> list[str]:
    """
    A key per vector (normalised to length 1), equal for vectors equal in
    their first embeddingKeyDecimals decimals, i.e. of a cosine similarity of
    1 up to rounding. -0.0 and 0.0 are the same.
    """
    rounded = np.round(vectors, embeddingKeyDecimals) + 0.0
    return [hashlib.sha1(row.tobytes()).hexdigest()[:20] for row in rounded]

def bestSimilarities(ids : list, vectors : np.ndarray, otherIDs : list,
        otherVectors : np.ndarray) -> np.ndarray:
    """
    For every vector the highest cosine similarity to an other vector of the
    same concept; NaN if its concept has none.
    """
    best = np.full(len(ids), np.nan)
    if len(ids) == 0 or len(otherIDs) == 0:
        return best
    rowsByConcept  = pd.Series(range(len(ids))).groupby(pd.Series(ids)).agg(list).to_dict()
    otherByConcept = pd.Series(range(len(otherIDs))).groupby(pd.Series(otherIDs)).agg(list).to_dict()
    for hpoID, rows in rowsByConcept.items():
        others = otherByConcept.get(hpoID)
        if others is None:
            continue
        best[rows] = (vectors[rows] @ otherVectors[others].T).max(axis = 1)
    return best

def embedModel(modelID : str) -> None:
    """
    Embeds and normalises the generated and gold synonyms of one model's run,
    unless they are up to date.
    """
    modelName = modelNameOf(modelID)
    inputFile = outputFileStage(finalStage, modelName)
    outputs   = [outputFileEmbedded(modelName), outputFileEmbeddedGold(modelName)]
    if not any(outdated(output, [inputFile, inputFileTask], embedOutputSettings) for output in outputs):
        l.log(f"The embedded synonyms of {quote(modelName)} are up to date.")
        return

    l.printHeader(f"Embedding the Synonyms ({modelName})")
    generated = readSynonyms(inputFile)
    data      = readTask()
    concepts  = set(taskConcepts())
    gold      = goldStandard(data)
    gold      = gold[gold[idColumn].isin(concepts)].reset_index(drop = True)
    labels    = labelsByConcept(data)
    # Marked again, so that files of earlier runs follow the same rule as
    # the gold standard.
    generated = markAsLabel(generated, labels)
    l.log(f"{len(generated.index)} generated synonyms of {generated[idColumn].nunique()} " \
        f"concepts, {len(gold.index)} gold synonyms of {gold[idColumn].nunique()} concepts.")

    with l.step("Normalising the synonyms"):
        generated[normalizedColumn] = normalizedKeys(generated[synonymColumn].tolist())
        gold[normalizedColumn]      = normalizedKeys(gold[synonymColumn].tolist())

    with l.step("Embedding the synonyms and labels"):
        generatedVectors = embed(generated[synonymColumn].tolist())
        goldVectors      = embed(gold[synonymColumn].tolist())
        generated[embeddingKeyColumn] = embeddingKeys(generatedVectors)
        gold[embeddingKeyColumn]      = embeddingKeys(goldVectors)

        # The similarity to the label of the synonyms whose stage has not
        # computed it (the LLM's).
        missing = generated[similarityColumn].isna().to_numpy()
        if missing.any():
            labelVectors = embed([labels.get(hpoID, "") for hpoID in generated.loc[missing, idColumn]])
            generated.loc[missing, similarityColumn] = np.sum(
                generatedVectors[missing] * labelVectors, axis = 1)

    with l.step("Comparing generated and gold synonyms"):
        generated[goldSimilarityColumn] = bestSimilarities(generated[idColumn].tolist(),
            generatedVectors, gold[idColumn].tolist(), goldVectors)
        # A synonym equal to its label is no synonym, so no gold synonym is
        # compared with it.
        for stage in experimentStages:
            ofStage = (generated[sourceColumn] == stage).to_numpy() & ~generated[asLabelColumn].to_numpy()
            gold[stageSimilarityColumn(stage)] = bestSimilarities(gold[idColumn].tolist(),
                goldVectors, generated.loc[ofStage, idColumn].tolist(), generatedVectors[ofStage])

    match = syngenConfig.evaluation.embedding_match
    for stage in experimentStages:
        ofStage = generated[sourceColumn] == stage
        l.log(f"{quote(stage)}: {ofStage.sum()} synonyms, mean similarity to the label " \
            f"{generated.loc[ofStage, similarityColumn].mean():.4f}; " \
            f"{(gold[stageSimilarityColumn(stage)] >= match - similarityTolerance).sum()} gold " \
            f"synonyms have one of a similarity of at least {match}.")

    writeHugeCSV(generated, outputFileEmbedded(modelName))
    writeSettings(outputFileEmbedded(modelName), embedOutputSettings)
    writeHugeCSV(gold, outputFileEmbeddedGold(modelName))
    writeSettings(outputFileEmbeddedGold(modelName), embedOutputSettings)

def main() -> None:
    """
    Embeds the synonyms of the experiment for every model whose output is
    missing or outdated.
    """
    startRun()
    runForModels("Embedding the synonyms", embedModel)


if __name__ == "__main__":
    main()
