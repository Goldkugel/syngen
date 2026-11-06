"""
Helpers shared by the scripts of syngen: the transformed HPO in the format of
the adapter package and its gold standard, the information of the concepts
for the prompts, the settings records that decide whether an output is
outdated, reading and writing CSV files, parsing the answers of the LLM, the
keys synonyms are compared by, the preparation run (transformation,
development and evaluation set, few-shot examples).
"""

import ast
import functools
import json
import os
import re
import sys
import textwrap
import unicodedata

import numpy        as np
import pandas       as pd

sys.dont_write_bytecode = True

from config         import *

from logger         import Logger, writeJSON

from adapter                    import BaseAdapterConfig, semanticClass, sourceType
from adapter                    import labelClass, definitionClass, commentClass, childrenClass
from adapter                    import exactSynonymClass
from adapter                    import synonymClass as synonymAttribute
from adapter.BaseAdapterUtils   import additionalSeparator, isFile
from adapter.BaseAdapterUtils   import writeHugeCSV as writeHugeCSVAtomically

# The transformed HPO is in the adapter format: one row per (id, attribute,
# value) with a dict of additional information, e.g. {"semantic_class":
# "exact", "source_type": "layperson"}. Column names from adapter.hpo.
hpoAdapterConfig = BaseAdapterConfig.model_validate(configurationFile["adapter"]["hpo"])
idColumn         = hpoAdapterConfig.id_column
attributeColumn  = hpoAdapterConfig.attribute_column
valueColumn      = hpoAdapterConfig.value_column
additionalColumn = hpoAdapterConfig.additional_column

transformedColumns = [idColumn, attributeColumn, valueColumn, additionalColumn]

def additionalValues(additional, key : str) -> list:
    """
    The values of a key of the additional information of a row; a synonym
    listed several times for a concept has several, e.g. ["exact", "related"]
    (see the adapter's mergeAdditional()). [] if the key is missing.
    """
    if not isinstance(additional, dict):
        return []
    values = [value.strip() for value in str(additional.get(key, "")).split(additionalSeparator)]
    return [value for value in values if value != ""]

# ------------------------------------------------------------------------------
# Logging and text helpers
# ------------------------------------------------------------------------------

def quote(string : str) -> str:
    """
    The string in quotation marks, e.g. for the prompts and the log.
    """
    return quotationCharacter + str(string) + quotationCharacter

def shorten(string, width : int = 60) -> str:
    """
    A text cut at a word to at most width characters, with " ..." where it
    is cut, so that a log row stays short.
    """
    return textwrap.shorten(str(string), width = width, placeholder = " ...")

def applyFormat(l : list[str] = []) -> str:
    """
    A list of strings as one text, each quoted and separated by commas, e.g.
    '"a", "b"'; a single string is only quoted, and an empty list gives
    "None".
    """
    string = "None"
    if len(l) > 0:
        if isinstance(l, str):
            string = quote(l)
        else:
            string = ", ".join(quote(element) for element in l)
    return string

# ------------------------------------------------------------------------------
# The settings an output was made with are stored next to it
# (<output>.settings.json), so that it is made again when they change, not
# only when one of its inputs is newer.
# ------------------------------------------------------------------------------

def isOutdated(output : str, inputs : list) -> bool:
    """
    Whether a file is missing or older than one of the files it is made from.
    """
    return not isFile(output) or any(
        os.path.getmtime(output) < os.path.getmtime(i) for i in inputs if isFile(i))

def settingsFile(output : str) -> str:
    """
    The file holding the settings an output was made with.
    """
    return output + ".settings.json"

def changedSettings(output : str, settings : dict) -> list:
    """
    The settings (top-level keys) that differ from those the output was made
    with; all of them if they were not recorded.
    """
    try:
        with open(settingsFile(output), "r", encoding = "utf-8") as f:
            recorded = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return list(settings.keys())

    # Through JSON, as the recorded settings were, e.g. tuples become lists.
    current = json.loads(json.dumps(settings))
    return [key for key in current if recorded.get(key) != current[key]]

def writeSettings(output : str, settings : dict) -> None:
    """
    Records the settings an output was made with, next to it (atomically).
    """
    writeJSON(settings, settingsFile(output))

def outdated(output : str, inputs : list, settings : dict, log : bool = True) -> bool:
    """
    Whether an output is missing, older than one of its inputs, or made with
    other settings; logs why, unless log is False.
    """
    if isOutdated(output, inputs):
        if log: Logger().log(f"{quote(os.path.basename(output))} is missing or older than " \
            f"{applyFormat([os.path.basename(i) for i in inputs])}.")
        return True

    changed = changedSettings(output, settings)
    if len(changed) > 0:
        if log: Logger().log(f"{quote(os.path.basename(output))} was made with other or unrecorded " \
            f"settings: {applyFormat(changed)}.")
        return True

    return False

# ------------------------------------------------------------------------------
# Files
# ------------------------------------------------------------------------------

def exitIfFileNotExist(file : str) -> None:
    """
    Terminate the program if the specified file does not exist.
    """
    if not isFile(file):
        Logger().error(f"{quote(os.path.basename(file))} could not be found. Terminating.")
        sys.exit(1)

def readCSV(file : str) -> pd.DataFrame:
    """
    Read a CSV file from disk with logging; empty cells stay empty strings
    in the text columns.
    """
    Logger().printReadFileStart(file)
    ret = pd.read_csv(file, low_memory = False, keep_default_na = False, na_values = [""])
    Logger().printReadFileEnd(file)
    return ret

def writeHugeCSV(data : pd.DataFrame, file : str) -> None:
    """
    Write a DataFrame atomically as a comma-separated CSV file (see the
    adapter's writeHugeCSV()), creating its folder if needed.
    """
    folder = os.path.dirname(file)
    if folder:
        os.makedirs(folder, exist_ok = True)

    writeHugeCSVAtomically(data, file, ",")

def readTransformed(file : str) -> pd.DataFrame:
    """
    Read a transformed HPO file, with the additional information as dicts.
    """
    ret = readCSV(file)
    ret[valueColumn] = ret[valueColumn].fillna("").astype(str)
    # Only few distinct dicts, so each is parsed once.
    entries = ret[additionalColumn].fillna("{}").astype(str)
    parsed  = {entry : ast.literal_eval(entry) for entry in entries.unique()}
    ret[additionalColumn] = entries.map(parsed)
    return ret

# ------------------------------------------------------------------------------
# Keys synonyms are compared by
# ------------------------------------------------------------------------------

_whitespace = re.compile(r"\s+")

def lexicalKey(text) -> str:
    """
    The key of the lexical match: the text in Unicode NFKC, case-folded, with
    every run of white space as one space, e.g. "Short  Stature" ->
    "short stature". Case is ignored, as LLMs and abstracts capitalise terms
    differently than the HPO.
    """
    return _whitespace.sub(" ", unicodedata.normalize("NFKC", str(text)).casefold()).strip()

@functools.lru_cache(maxsize = 1)
def normalizer():
    """
    The Normalizer (normalizer section of the configuration), loaded once.
    """
    from normalizer import Normalizer
    return Normalizer(configurationFile)

def normalizedKeys(texts : list) -> list[str]:
    """
    The keys of the normalised match: the tokens of the normalizer (lower
    case, lemmatised, without stop words), sorted and joined by spaces, so
    that word order and inflection do not matter, e.g. "Hypoplasia of the
    radius" and "radius hypoplasia" -> "hypoplasia radius". "" for a text
    without tokens, which matches nothing.
    """
    texts = [str(text) for text in texts]
    unique = list(dict.fromkeys(texts))
    tokens = normalizer().normalize_batch(unique) if len(unique) > 0 else []
    keys = {text : " ".join(sorted(t)) for text, t in zip(unique, tokens)}
    return [keys[text] for text in texts]

# Embedder settings that do not change the vectors.
embedderSettingsWithoutEffect = ["device", "backend", "num_threads",
    "batch_size", "cache_folder", "cache_decimals", "progress_interval"]
embedSettings = {
    key : value for key, value in configurationFile["embedder"].items()
    if key not in embedderSettingsWithoutEffect
}

@functools.lru_cache(maxsize = 1)
def embedder():
    """
    The Embedder of the similarities (embedder section of the configuration,
    only its first model, embeddingModelID), loaded once.
    """
    from embedder import Embedder

    configuration = json.loads(json.dumps(configurationFile))
    configuration["embedder"]["model_id"] = [embeddingModelID]
    return Embedder(configuration)

def embed(texts : list) -> np.ndarray:
    """
    The vectors of texts with the embedding model, normalised to length 1, so
    that the dot product of two is their cosine similarity. A text is encoded
    once, however often it is given.
    """
    texts  = [str(text) for text in texts]
    unique = list(dict.fromkeys(texts))
    if len(unique) == 0:
        return np.zeros((0, 0))
    vectors = np.asarray(embedder().embed(unique)[embeddingModelID], dtype = np.float32)
    norms   = np.linalg.norm(vectors, axis = 1, keepdims = True)
    vectors = vectors / np.where(norms > 0, norms, 1)
    index   = {text : i for i, text in enumerate(unique)}
    return vectors[[index[text] for text in texts]]

def similarities(texts : list, references : list) -> np.ndarray:
    """
    The cosine similarity of every text to the reference at the same
    position, e.g. of synonyms to the labels of their concepts.
    """
    if len(texts) == 0:
        return np.zeros(0)
    vectors = embed(list(texts) + list(references))
    a, b = vectors[:len(texts)], vectors[len(texts):]
    return np.sum(a * b, axis = 1)

# ------------------------------------------------------------------------------
# Concepts, their information and the gold standard
# ------------------------------------------------------------------------------

def labelsByConcept(data : pd.DataFrame) -> dict:
    """
    The label of every concept of a transformed HPO file (the first, should
    a concept have several).
    """
    labels = data[data[attributeColumn] == labelClass]
    return labels.drop_duplicates(subset = idColumn).set_index(idColumn)[valueColumn].to_dict()

def conceptInformation(data : pd.DataFrame) -> dict:
    """
    Per concept of a transformed HPO file: its label, definition, comment and
    the labels of its parents and children, as the prompts use them, e.g.
    {"HP:0001250": {"label": "Seizure", "definition": "...", "comment": "",
    "parents": [...], "children": [...]}}. Concepts without label are left
    out.
    """
    def joined(attribute : str) -> dict:
        rows = data[data[attributeColumn] == attribute]
        return rows.groupby(idColumn, sort = False)[valueColumn].agg(
            lambda values: " ".join(str(v) for v in values)).to_dict()

    labels      = labelsByConcept(data)
    definitions = joined(definitionClass)
    comments    = joined(commentClass)

    # A child row reads "id has child value".
    relations = data[data[attributeColumn] == childrenClass]
    children  = relations.groupby(idColumn, sort = False)[valueColumn].agg(list).to_dict()
    parents   = relations.groupby(valueColumn, sort = False)[idColumn].agg(list).to_dict()

    def relatedLabels(related : dict, hpoID : str) -> list:
        return [labels[i] for i in related.get(hpoID, []) if i in labels]

    return {
        hpoID : {
            "label"      : label,
            "definition" : definitions.get(hpoID, ""),
            "comment"    : comments.get(hpoID, ""),
            "parents"    : relatedLabels(parents, hpoID),
            "children"   : relatedLabels(children, hpoID)
        }
        for hpoID, label in labels.items()
    }

# The columns of the gold standard: concept and gold synonym.
goldColumns = [idColumn, synonymColumn]

def goldStandard(data : pd.DataFrame) -> pd.DataFrame:
    """
    The gold synonyms of a transformed HPO file: the synonyms HPO lists only
    as exact (one listed also as related, broad or narrow is left out), not
    of a type of syngen.gold.exclude_types, and not equal to the label of
    their concept (lexicalKey()). One row per concept and synonym, with the
    columns of goldColumns.
    """
    excluded = set(syngenConfig.gold.exclude_types)
    synonyms = data[data[attributeColumn] == synonymAttribute]
    onlyExact = synonyms[additionalColumn].map(
        lambda a: set(additionalValues(a, semanticClass)) == {exactSynonymClass})
    allowedType = synonyms[additionalColumn].map(
        lambda a: len(set(additionalValues(a, sourceType)) & excluded) == 0)
    gold = synonyms[onlyExact & allowedType]

    labels = {hpoID : lexicalKey(label) for hpoID, label in labelsByConcept(data).items()}
    asLabel = [lexicalKey(value) == labels.get(hpoID) for hpoID, value
        in zip(gold[idColumn], gold[valueColumn])]
    gold = gold[~np.array(asLabel, dtype = bool)]

    gold = gold.rename(columns = {valueColumn : synonymColumn})[goldColumns]
    # A synonym listed twice for a concept is one gold synonym.
    keys = gold[synonymColumn].map(lexicalKey)
    return gold[~pd.Series(list(zip(gold[idColumn], keys)), index = gold.index).duplicated()] \
        .reset_index(drop = True)

def otherSynonyms(data : pd.DataFrame) -> pd.DataFrame:
    """
    The synonyms of a transformed HPO file that are not gold synonyms
    (related, broad, narrow, ...), with their classes joined by ", " in the
    source column, to tell which generated synonyms HPO lists otherwise.
    """
    synonyms = data[data[attributeColumn] == synonymAttribute]
    gold     = goldStandard(data)
    goldKeys = set(zip(gold[idColumn], gold[synonymColumn].map(lexicalKey)))
    keys     = list(zip(synonyms[idColumn], synonyms[valueColumn].map(lexicalKey)))
    other    = synonyms[[key not in goldKeys for key in keys]]
    return pd.DataFrame({
        idColumn      : other[idColumn].to_numpy(),
        synonymColumn : other[valueColumn].to_numpy(),
        sourceColumn  : other[additionalColumn].map(
            lambda a: ", ".join(additionalValues(a, semanticClass))).to_numpy()
    })

def logGoldCounts(name : str, data : pd.DataFrame) -> None:
    """
    Logs how many concepts and gold synonyms a transformed HPO file holds.
    """
    gold = goldStandard(data)
    Logger().log(f"{name}: {data[idColumn].nunique()} concepts, {len(gold.index)} gold " \
        f"synonyms of {gold[idColumn].nunique()} concepts.")

# ------------------------------------------------------------------------------
# Parsing the answers of the LLM
# ------------------------------------------------------------------------------

def parseTerms(answer, key : str) -> list[str] | None:
    """
    The terms of an answer parsed by model.toJSON(): the strings of the list
    under `key` of a JSON object, or of a JSON list itself. Empty strings and
    other values are dropped. None if the answer has no such list.
    """
    if isinstance(answer, dict):
        answer = answer.get(key)
    if not isinstance(answer, list):
        return None
    terms = [str(term).strip() for term in answer if isinstance(term, (str, int, float))
        and not isinstance(term, bool)]
    return [term for term in terms if term != ""]

# ------------------------------------------------------------------------------
# The synonym files of the stages
# ------------------------------------------------------------------------------

# Every stage file has these columns; a column a stage does not fill is empty.
synonymFileColumns = [idColumn, synonymColumn, sourceColumn, modelColumn, roundsColumn,
    firstRoundColumn, seedColumn, evidenceColumn, frequencyColumn, similarityColumn,
    asLabelColumn]

def synonymFrame(rows : list[dict]) -> pd.DataFrame:
    """
    A synonym file from rows (dicts of some of synonymFileColumns).
    """
    frame = pd.DataFrame(rows, columns = synonymFileColumns)
    frame[asLabelColumn] = frame[asLabelColumn].astype("boolean").fillna(False).astype(bool)
    return frame

def readSynonyms(file : str) -> pd.DataFrame:
    """
    A synonym file of a stage.
    """
    exitIfFileNotExist(file)
    data = readCSV(file)
    data[synonymColumn] = data[synonymColumn].fillna("").astype(str)
    data[asLabelColumn] = data[asLabelColumn].astype(str).str.lower() == "true"
    return data

def rankCandidates(candidates : pd.DataFrame, previous : pd.DataFrame, labels : dict,
        minSimilarity : float, maxCandidates : int) -> pd.DataFrame:
    """
    The candidates of a stage (rows of a synonym file) that become its
    synonyms: per concept, those not yet among the synonyms of the stages
    before (`previous`) and not equal to the label (lexicalKey()), with a
    cosine similarity to the label of at least minSimilarity, the most
    similar maxCandidates of them. Fills the similarity column.
    """
    l = Logger()
    if len(candidates.index) == 0:
        return candidates

    keys     = candidates[synonymColumn].map(lexicalKey)
    known    = set(zip(previous[idColumn], previous[synonymColumn].map(lexicalKey)))
    labelKey = candidates[idColumn].map(lambda hpoID: lexicalKey(labels.get(hpoID, "")))
    new      = [(hpoID, key) not in known and key != label and key != ""
        for hpoID, key, label in zip(candidates[idColumn], keys, labelKey)]
    candidates = candidates[new].copy()
    l.log(f"{len(candidates.index)} candidates are new (not the label nor a synonym of " \
        "the stages before).")

    candidates[similarityColumn] = similarities(candidates[synonymColumn].tolist(),
        [labels.get(hpoID, "") for hpoID in candidates[idColumn]])
    similar = candidates[candidates[similarityColumn] >= minSimilarity]
    l.log(f"{len(similar.index)} candidates have a similarity to their label of at " \
        f"least {minSimilarity}.")

    ranked = similar.sort_values([idColumn, similarityColumn], ascending = [True, False],
        kind = "stable").groupby(idColumn, sort = False).head(maxCandidates)
    l.log(f"Kept {len(ranked.index)} candidates of {ranked[idColumn].nunique()} concepts, " \
        f"at most {maxCandidates} per concept.")
    return ranked.reset_index(drop = True)

def markAsLabel(synonyms : pd.DataFrame, labels : dict) -> pd.DataFrame:
    """
    Sets the as_label column: whether a synonym equals the label of its
    concept, lexically or normalised.
    """
    labelKeys      = {hpoID : lexicalKey(label) for hpoID, label in labels.items()}
    labelNormKeys  = dict(zip(labels.keys(), normalizedKeys(list(labels.values()))))
    synonymNormKeys = normalizedKeys(synonyms[synonymColumn].tolist())
    synonyms[asLabelColumn] = [
        lexicalKey(synonym) == labelKeys.get(hpoID) or
            (key != "" and key == labelNormKeys.get(hpoID))
        for hpoID, synonym, key in zip(synonyms[idColumn], synonyms[synonymColumn], synonymNormKeys)
    ]
    return synonyms

# ------------------------------------------------------------------------------
# The preparation run: the transformed HPO, the development and evaluation
# set and the few-shot examples.
# ------------------------------------------------------------------------------

# The settings an output depends on (see outdated()).
transformSettings = {
    "adapter" : configurationFile["adapter"],
    "format"  : transformedColumns
}

def transformOutdated(log : bool = True) -> bool:
    """
    Whether hpo.csv is missing, older than hp.owl or made with other
    settings; the reason is logged if `log`.
    """
    return outdated(outputFileTransformedFull, [inputFileTransformed], transformSettings, log)

def transformHPO() -> None:
    """
    Transforms hp.owl with the adapter into hpo.csv (adapter format), with all
    synonyms: the gold standard is taken from it by goldStandard(). The
    adapter keeps only the concepts that are not obsolete.
    """
    from adapter import HPOAdapter

    l = Logger()
    l.printHeader("Transforming the Raw HPO Data")

    exitIfFileNotExist(inputFileTransformed)
    adapter = HPOAdapter(configurationFile)
    adapter.load()
    data = adapter.data

    l.printRowCount(data)
    for attribute, count in data[attributeColumn].value_counts().items():
        l.log(f"{quote(attribute)}: {count} rows.")
    logGoldCounts("Transformed HPO", data)

    writeHugeCSV(data, outputFileTransformedFull)
    writeSettings(outputFileTransformedFull, transformSettings)

developmentSetSettings = {
    "development_set" : syngenConfig.development_set.model_dump(),
    "gold"            : syngenConfig.gold.model_dump(),
    "format"          : transformedColumns
}

def developmentSetOutdated(log : bool = True) -> bool:
    """
    Whether the development or evaluation set or the few-shot examples are
    missing, older than hpo.csv or drawn with other settings; the reason is
    logged if `log`.
    """
    return any([outdated(output, [outputFileTransformedFull], developmentSetSettings, log)
        for output in [outputFileTransformedDevelopment, outputFileTransformedEvaluation,
            outputFileFewShot]])

def writeDevelopmentSet() -> pd.DataFrame:
    """
    Draws concepts with gold synonyms at random from hpo.csv until their gold
    synonyms number at least syngen.development_set.synonym_count (the last
    concept is taken whole), and writes their rows (label, definition,
    comment, synonyms, child relations in both directions) and the labels of
    their parents and children to hpo.development.csv. Writes hpo.csv without
    the synonyms of these concepts to hpo.evaluation.csv: their labels stay,
    as other concepts name them as parents and children. Draws the few-shot
    examples from the development concepts with at least two gold synonyms
    (hpo.fewshot.json). Records the settings next to all three.

    :return: The development set.
    """
    l = Logger()
    settings = syngenConfig.development_set

    exitIfFileNotExist(outputFileTransformedFull)
    data = readTransformed(outputFileTransformedFull)
    gold = goldStandard(data)

    counts   = gold.groupby(idColumn).size()
    # In the order of hpo.csv, then shuffled with the seed.
    concepts = counts.index.to_series().sample(frac = 1,
        random_state = np.random.RandomState(settings.seed % (2 ** 32)))
    cumulative = counts[concepts.to_numpy()].cumsum()
    if cumulative.iloc[-1] < settings.synonym_count:
        l.error(f"Only {cumulative.iloc[-1]} gold synonyms, fewer than " \
            f"syngen.development_set.synonym_count ({settings.synonym_count}).")
        sys.exit(1)
    # Up to and including the concept that reaches the count.
    last  = int(np.argmax(cumulative.to_numpy() >= settings.synonym_count))
    drawn = list(cumulative.index[:last + 1])
    drawnSet = set(drawn)
    l.log(f"Drawing {len(drawn)} of {len(counts)} concepts with gold synonyms " \
        f"(seed {settings.seed}): {cumulative.iloc[last]} gold synonyms.")

    attribute    = data[attributeColumn]
    conceptRows  = data[idColumn].isin(drawnSet) & (attribute != childrenClass)
    relationRows = (attribute == childrenClass) & (
        data[idColumn].isin(drawnSet) | data[valueColumn].isin(drawnSet))
    related      = (set(data.loc[relationRows, idColumn]) |
        set(data.loc[relationRows, valueColumn])) - drawnSet
    relatedLabelRows = data[idColumn].isin(related) & (attribute == labelClass)

    development = data[conceptRows | relationRows | relatedLabelRows].reset_index(drop = True)
    l.log(f"Development set: {len(development.index)} rows, {relationRows.sum()} relations, " \
        f"{len(related)} labels of parents and children.")
    logGoldCounts("Development set", development)

    developmentSynonyms = data[idColumn].isin(drawnSet) & (attribute == synonymAttribute)
    evaluation = data[~developmentSynonyms].reset_index(drop = True)
    l.log(f"Evaluation set: {len(evaluation.index)} of the {len(data.index)} rows of " \
        f"{quote(os.path.basename(outputFileTransformedFull))}.")
    logGoldCounts("Evaluation set", evaluation)

    # The few-shot examples: concepts with at least two gold synonyms, so an
    # example shows that a concept can have several.
    candidates = [hpoID for hpoID in drawn if counts[hpoID] >= 2]
    if len(candidates) < settings.few_shot_count:
        l.error(f"Only {len(candidates)} development concepts with two gold synonyms, " \
            f"fewer than syngen.development_set.few_shot_count ({settings.few_shot_count}).")
        sys.exit(1)
    information = conceptInformation(development)
    goldByConcept = gold.groupby(idColumn)[synonymColumn].agg(list).to_dict()
    # The first ones in the drawn (random) order.
    fewShotExamples = [
        {"id" : hpoID} | information[hpoID] | {"synonyms" : goldByConcept[hpoID]}
        for hpoID in candidates[:settings.few_shot_count]
    ]
    l.log(f"Few-shot examples: {applyFormat([e['id'] for e in fewShotExamples])}.")

    writeHugeCSV(development, outputFileTransformedDevelopment)
    writeSettings(outputFileTransformedDevelopment, developmentSetSettings)
    writeHugeCSV(evaluation, outputFileTransformedEvaluation)
    writeSettings(outputFileTransformedEvaluation, developmentSetSettings)
    writeJSON(fewShotExamples, outputFileFewShot)
    writeSettings(outputFileFewShot, developmentSetSettings)
    return development

def prepare() -> None:
    """
    The preparation: brings the transformed HPO, the development and
    evaluation set and the few-shot examples up to date. Only the first of the
    runs started at the same time makes them; the others wait for the lock.
    The reason is logged only once the lock is taken.
    """
    from logger import lockedFile

    l = Logger()
    if transformOutdated(log = False):
        with lockedFile(outputFileTransformedFull):
            if transformOutdated():
                with l.step("Transforming the HPO"):
                    transformHPO()
    else:
        l.log("The transformed HPO is up to date.")

    if developmentSetOutdated(log = False):
        with lockedFile(outputFileTransformedDevelopment):
            if developmentSetOutdated():
                with l.step("Drawing the development and evaluation set"):
                    writeDevelopmentSet()
    else:
        l.log("The development and evaluation set are up to date.")

# ------------------------------------------------------------------------------
# The concepts of a run
# ------------------------------------------------------------------------------

def readFewShotExamples() -> list[dict]:
    """
    The few-shot examples drawn by writeDevelopmentSet().
    """
    exitIfFileNotExist(outputFileFewShot)
    with open(outputFileFewShot, "r", encoding = "utf-8") as f:
        return json.load(f)

def developmentConcepts() -> set:
    """
    The concepts drawn for the development set: those with synonym rows in
    hpo.development.csv (the others only give the labels of parents and
    children).
    """
    exitIfFileNotExist(outputFileTransformedDevelopment)
    development = readCSV(outputFileTransformedDevelopment)
    return set(development.loc[development[attributeColumn] == synonymAttribute, idColumn])

@functools.lru_cache(maxsize = 1)
def readTask() -> pd.DataFrame:
    """
    The transformed HPO file of the experiment (inputFileTask), read once per
    run. The callers must not change the returned data.
    """
    exitIfFileNotExist(inputFileTask)
    return readTransformed(inputFileTask)

@functools.lru_cache(maxsize = 1)
def taskConcepts() -> tuple:
    """
    The concepts the experiment generates synonyms for, in the order of the
    file: with "test" the development concepts, otherwise all others; the
    few-shot concepts never.
    """
    data        = readTask()
    development = developmentConcepts()
    fewShotIDs  = {example["id"] for example in readFewShotExamples()}
    labelled    = data.loc[data[attributeColumn] == labelClass, idColumn].drop_duplicates()
    if reduceToTestIDs:
        concepts = [hpoID for hpoID in labelled if hpoID in development]
    else:
        concepts = [hpoID for hpoID in labelled if hpoID not in development]
    return tuple(hpoID for hpoID in concepts if hpoID not in fewShotIDs)
