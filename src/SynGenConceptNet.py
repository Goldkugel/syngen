"""
The ConceptNet stage of syngen: replaces the words of the label and of the
synonyms of the stages before (syngen.conceptnet.seed_sources) by their
neighbours in ConceptNet (syngen.conceptnet.relations, e.g. Synonym and
FormOf), optionally lets the LLM keep only the neighbours that can replace a
word in the concept, builds new terms with up to max_substitutions
replacements, and keeps per concept the terms most similar to its label.
Writes the synonyms of the stages before together with its own.

Ported from the student project ba-syngen-conceptnet-llm, with two changes:
inflections and spellings are not generated with the SPECIALIST Lexicon
(LVG), as the normalised match (normalizer package) ignores them anyway, and
the label is not rewritten structurally (e.g. "X of Y" -> "Y X"), as the
normalised match ignores word order and stop words.

    python3 ./src/SynGenConceptNet.py <index> [<model>]

The database is built once by SynGenConceptNetDB.py.
"""

import itertools
import json
import logging
import shlex
import sqlite3
import sys

import pandas as pd

sys.dont_write_bytecode = True

from SynGenUtils         import *
from config              import *

from SynGenConceptNetDB  import databaseSettings
from logger              import Logger

l = Logger(config = configFile)

conceptnetConfig = syngenConfig.conceptnet

# Index of the filter template in llm.prompt_files.
filterPromptIndex = 1

# The parts of speech (of the normalizer's spaCy model) a replaced span starts
# and ends with.
contentPOS = {"NOUN", "PROPN", "ADJ", "VERB", "ADV"}

# ==============================================================================
# ConceptNet
# ==============================================================================

class ConceptNet:
    """
    The neighbours of words in the ConceptNet database: the words reached
    from a word along the configured relations, in both directions, within
    syngen.conceptnet.hops steps, in the first configured language, without
    concepts of a blacklisted sense. Looked up once per word.
    """

    def __init__(self, path : str) -> None:
        exitIfFileNotExist(path)
        self.connection = sqlite3.connect(f"file:{path}?mode=ro", uri = True)
        recorded = self.connection.execute("SELECT settings FROM settings").fetchone()[0]
        if json.loads(recorded) != json.loads(databaseSettings()):
            raise ValueError(f"{quote(path)} was built with other relations or languages " \
                f"({recorded}); build it again with SynGenConceptNetDB.py.")
        self.language    = conceptnetConfig.languages[0]
        self.blacklisted = {sense.lower() for sense in conceptnetConfig.blacklisted_senses}
        self.known       = {}
        self.cache       = {}

    def close(self) -> None:
        self.connection.close()

    def concepts(self, word : str, language : str | None = None) -> list[int]:
        """The concepts spelled `word` (in `language`), without blacklisted senses."""
        query = "SELECT id, sense FROM concepts WHERE word = ?"
        parameters = [word]
        if language is not None:
            query += " AND language = ?"
            parameters.append(language)
        return [i for i, sense in self.connection.execute(query, parameters)
            if sense.lower() not in self.blacklisted]

    def contains(self, word : str) -> bool:
        """Whether the word is a concept of the first language."""
        if word not in self.known:
            self.known[word] = len(self.concepts(word, self.language)) > 0
        return self.known[word]

    def neighbours(self, word : str) -> list[str]:
        """
        The words reached from `word` within syngen.conceptnet.hops steps
        (without `word` itself), in lower case.
        """
        if word in self.cache:
            return self.cache[word]

        reached  = set(self.concepts(word, self.language))
        frontier = set(reached)
        for _ in range(conceptnetConfig.hops):
            if len(frontier) == 0:
                break
            found = set()
            for chunk in chunks(list(frontier), 500):
                marks = ",".join("?" * len(chunk))
                found.update(i for (i,) in self.connection.execute(
                    f"SELECT target FROM edges WHERE source IN ({marks})", chunk))
                found.update(i for (i,) in self.connection.execute(
                    f"SELECT source FROM edges WHERE target IN ({marks})", chunk))
            frontier = found - reached
            reached |= found

        words = set()
        for chunk in chunks(list(reached), 500):
            marks = ",".join("?" * len(chunk))
            for neighbour, language, sense in self.connection.execute(
                    f"SELECT word, language, sense FROM concepts WHERE id IN ({marks})", chunk):
                if language == self.language and sense.lower() not in self.blacklisted:
                    words.add(neighbour)
        words.discard(word)
        self.cache[word] = sorted(words)
        return self.cache[word]

def chunks(values : list, size : int):
    """The values in lists of at most `size`."""
    for start in range(0, len(values), size):
        yield values[start:start + size]

# ==============================================================================
# The terms whose words are replaced, and their spans.
# ==============================================================================

def seedTerms(previous : pd.DataFrame, labels : dict, concepts : list) -> dict:
    """
    Per concept the terms whose words are replaced: its label (if "label" is
    a seed source) and its synonyms of the other seed sources (not those equal
    to the label), each once (lexicalKey()).
    """
    sources = set(conceptnetConfig.seed_sources)
    usable  = previous[previous[sourceColumn].isin(sources) & ~previous[asLabelColumn]]
    byConcept = usable.groupby(idColumn, sort = False)[synonymColumn].agg(list).to_dict()
    seeds = {}
    for hpoID in concepts:
        terms = ([labels[hpoID]] if "label" in sources else []) + byConcept.get(hpoID, [])
        unique = {}
        for term in terms:
            unique.setdefault(lexicalKey(term), term)
        seeds[hpoID] = [term for key, term in unique.items() if key != ""]
    return seeds

def spansOf(terms : list, conceptnet : ConceptNet) -> dict:
    """
    Per term its spans (start, end, text in lower case): the runs of at most
    syngen.conceptnet.max_span tokens that start and end with a content word
    and are a ConceptNet concept, e.g. "short", "stature" and "short stature"
    in "Short stature".
    """
    nlp = normalizer().nlp
    spans = {}
    for term, doc in zip(terms, nlp.pipe(terms, batch_size = 256)):
        found = []
        tokens = list(doc)
        for i, first in enumerate(tokens):
            if first.pos_ not in contentPOS:
                continue
            for j in range(i, min(i + conceptnetConfig.max_span, len(tokens))):
                if tokens[j].pos_ not in contentPOS:
                    continue
                start, end = first.idx, tokens[j].idx + len(tokens[j].text)
                text = term[start:end].lower()
                if conceptnet.contains(text):
                    found.append((start, end, text))
        spans[term] = found
    return spans

# ==============================================================================
# The filter: the LLM keeps the neighbours that can replace a span.
# ==============================================================================

def filterNeighbours(questions : list, information : dict, modelID : str) -> dict:
    """
    Asks the LLM per (concept, span, term) which neighbours can replace the
    span in the concept, syngen.conceptnet.filter_chunk neighbours per
    prompt, with syngen.conceptnet.filter_temperature. Returns the kept
    neighbours per (concept, span); a neighbour the answer does not name, or
    of an answer without a list, is dropped.
    """
    from model import Model

    logging.getLogger("vllm").setLevel(logging.ERROR)

    prompts = []
    for hpoID, span, term, neighbours in questions:
        info = information[hpoID]
        details = f"\n* Definition: {quote(info['definition'])}" if info["definition"] else ""
        for chunk in chunks(neighbours, conceptnetConfig.filter_chunk):
            prompts.append(((hpoID, span), chunk, {
                "hpo_label"   : quote(info["label"]),
                "hpo_details" : details,
                "term"        : quote(term),
                "token"       : quote(span),
                "candidates"  : "\n".join(f"* {neighbour}" for neighbour in chunk)
            }))
    l.log(f"{len(prompts)} filter prompts for {len(questions)} words.")

    kept = {}
    if len(prompts) == 0:
        return kept

    # The filter decides, so it samples (almost) greedily.
    configuration = json.loads(json.dumps(configurationFile))
    configuration["llm"]["temperature"] = conceptnetConfig.filter_temperature
    modelIndex = configuration["llm"]["model_id"].index(modelID)

    with Model(configuration, modelIndex) as model:
        model.setPostfix(f"{generationStr}_{stagesStr(conceptnetStage)}_filter")
        model.addPromptsFromFile(filterPromptIndex, [values for _, _, values in prompts])
        model.generate()

        hallucinated, invalid = 0, 0
        for index, (key, chunk, _) in enumerate(prompts):
            answer = parseTerms(model.toJSON(index), keepKey)
            if answer is None:
                invalid += 1
                continue
            chunkKeys = {lexicalKey(neighbour) : neighbour for neighbour in chunk}
            for term in answer:
                if lexicalKey(term) in chunkKeys:
                    kept.setdefault(key, set()).add(chunkKeys[lexicalKey(term)])
                else:
                    hallucinated += 1
        model.logPrompts()

    l.log(f"Filter: {invalid} of {len(prompts)} answers without a list, {hallucinated} " \
        f"kept words that were no candidate, {sum(len(v) for v in kept.values())} kept.")
    return kept

# ==============================================================================
# Recombination: new terms with up to max_substitutions replaced spans.
# ==============================================================================

def recombine(term : str, replacements : list) -> list[tuple[str, str]]:
    """
    The new terms of a term and how they were made: every replacement of one
    span, then of two non-overlapping spans, and so on up to
    syngen.conceptnet.max_substitutions, at most
    syngen.conceptnet.max_terms_per_seed. `replacements` holds per span
    (start, end, text, neighbours).
    """
    results = []
    for count in range(1, conceptnetConfig.max_substitutions + 1):
        for combination in itertools.combinations(replacements, count):
            ordered = sorted(combination, key = lambda r: r[0])
            if any(a[1] > b[0] for a, b in zip(ordered, ordered[1:])):
                continue
            for choice in itertools.product(*[r[3] for r in ordered]):
                new, position = "", 0
                for (start, end, _, _), neighbour in zip(ordered, choice):
                    # ConceptNet's words are in lower case; a replaced word
                    # keeps the capital letter of the original.
                    if term[start].isupper() and neighbour[:1].islower():
                        neighbour = neighbour[0].upper() + neighbour[1:]
                    new += term[position:start] + neighbour
                    position = end
                new += term[position:]
                how = "; ".join(f"{r[2]} -> {n}" for r, n in zip(ordered, choice))
                results.append((new, how))
                if len(results) >= conceptnetConfig.max_terms_per_seed:
                    return results
    return results

# ==============================================================================
# The steps of this run.
# ==============================================================================

def runModel(modelID : str) -> None:
    """
    The ConceptNet stage of the synonyms one LLM generated.
    """
    modelName = modelNameOf(modelID)
    l.printHeader(f"Building Synonyms with ConceptNet ({modelName})")

    previous    = readSynonyms(outputFileStage(previousStage(conceptnetStage), modelName))
    information = conceptInformation(readTask())
    labels      = {hpoID : info["label"] for hpoID, info in information.items()}
    concepts    = [hpoID for hpoID in taskConcepts() if hpoID in labels]
    seeds       = seedTerms(previous, labels, concepts)
    terms       = list(dict.fromkeys(term for t in seeds.values() for term in t))
    l.log(f"{len(terms)} terms of {len(seeds)} concepts, seed sources " \
        f"{applyFormat(conceptnetConfig.seed_sources)}.")

    conceptnet = ConceptNet(conceptnetConfig.database)
    try:
        with l.step("Finding the spans and their neighbours"):
            spans = spansOf(terms, conceptnet)
            neighbours = {text : conceptnet.neighbours(text)
                for found in spans.values() for _, _, text in found}
    finally:
        conceptnet.close()
    l.log(f"{sum(len(s) for s in spans.values())} spans, {len(neighbours)} distinct; " \
        f"{sum(len(n) for n in neighbours.values())} neighbours.")

    # Per concept and span: the neighbours that may replace it.
    allowed = {}
    for hpoID, conceptTerms in seeds.items():
        for term in conceptTerms:
            for _, _, text in spans[term]:
                allowed.setdefault((hpoID, text), (term, neighbours[text]))
    if conceptnetConfig.filter:
        with l.step("Filtering the neighbours with the LLM"):
            questions = [(hpoID, text, term, words) for (hpoID, text), (term, words)
                in allowed.items() if len(words) > 0]
            kept = filterNeighbours(questions, information, modelID)
        allowed = {key : sorted(kept.get(key, set())) for key in allowed}
    else:
        allowed = {key : words for key, (_, words) in allowed.items()}

    with l.step("Building the new terms"):
        found = {}
        for hpoID, conceptTerms in seeds.items():
            for term in conceptTerms:
                replacements = [(start, end, text, allowed[(hpoID, text)])
                    for start, end, text in spans[term] if len(allowed[(hpoID, text)]) > 0]
                for new, how in recombine(term, replacements):
                    entry = found.setdefault((hpoID, lexicalKey(new)), {
                        idColumn : hpoID, synonymColumn : new, sourceColumn : conceptnetStage,
                        seedColumn : term, evidenceColumn : how, frequencyColumn : 0})
                    entry[frequencyColumn] += 1
        candidates = synonymFrame(list(found.values()))
        l.log(f"{len(candidates.index)} new terms of {candidates[idColumn].nunique()} concepts.")

    with l.step("Ranking the new terms"):
        ranked = rankCandidates(candidates, previous, labels,
            conceptnetConfig.min_similarity, conceptnetConfig.max_candidates)
    ranked[modelColumn] = modelName
    ranked = markAsLabel(ranked, labels)

    synonyms = pd.concat([previous, ranked], ignore_index = True)[synonymFileColumns]
    writeHugeCSV(synonyms, outputFileStage(conceptnetStage, modelName))
    l.log(f"{len(ranked.index)} synonyms from ConceptNet, {len(synonyms.index)} in total.")

def main() -> None:
    """
    Runs the ConceptNet stage of the experiment for every model.
    """
    l.log(f"Command: python3 {shlex.join(sys.argv)}")

    if experimentIndex is None:
        sys.exit("Give the index of the experiment.")
    if conceptnetStage not in experimentStages:
        l.log(f"Experiment {experimentIndex} has no ConceptNet stage.")
        return

    failed = []
    for modelID in modelIDs:
        try:
            with l.step(f"ConceptNet stage ({modelNameOf(modelID)})"):
                runModel(modelID)
        except (Exception, SystemExit):
            l.exception(f"The ConceptNet stage of {quote(modelID)} failed.", cmdline = False)
            failed.append(modelID)

    if len(failed) > 0:
        l.error(f"{len(failed)} of {len(modelIDs)} model(s) failed: {applyFormat(failed)}.")
        sys.exit(1)


# vLLM starts its workers with "spawn", which imports this script again: the
# steps must only run in the main process.
if __name__ == "__main__":
    main()
