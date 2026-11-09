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

The spans are found with the normalizer's spaCy model and the filter runs the
LLM, which need different Python environments; the stage therefore runs in
three steps, which hand their results on in files:

    python3 ./src/SynGenConceptNet.py <index> [<model>] --spans    normalizer
    python3 ./src/SynGenConceptNet.py <index> [<model>] --filter   model
    python3 ./src/SynGenConceptNet.py <index> [<model>] --build    normalizer

--filter does nothing if syngen.conceptnet.filter is off. The database is
built once by SynGenConceptNetDB.py.
"""

import itertools
import json
import os
import sqlite3
import sys

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

# Rows per SQL query with IN (SQLite allows 999 parameters in old versions).
queryChunk = 500

# ==============================================================================
# ConceptNet
# ==============================================================================

class ConceptNet:
    """
    The neighbours of words in the ConceptNet database: the words reached
    from a word along the configured relations, in both directions, within
    syngen.conceptnet.hops steps, in the first configured language. Concepts
    of a blacklisted sense are neither returned nor walked through. Looked up
    once per word.
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

    def concepts(self, word : str) -> list[int]:
        """The concepts spelled `word` in the first language, without blacklisted senses."""
        return [i for i, sense in self.connection.execute(
            "SELECT id, sense FROM concepts WHERE word = ? AND language = ?", (word, self.language))
            if sense.lower() not in self.blacklisted]

    def contains(self, word : str) -> bool:
        """Whether the word is a concept of the first language."""
        if word not in self.known:
            self.known[word] = len(self.concepts(word)) > 0
        return self.known[word]

    def neighbours(self, word : str) -> list[str]:
        """
        The words reached from `word` within syngen.conceptnet.hops steps
        (without `word` itself), in lower case and alphabetical order.
        """
        if word in self.cache:
            return self.cache[word]

        reached  = set(self.concepts(word))
        frontier = set(reached)
        words    = set()
        for _ in range(conceptnetConfig.hops):
            if len(frontier) == 0:
                break
            found = set()
            for chunk in chunks(list(frontier), queryChunk):
                marks = ",".join("?" * len(chunk))
                found.update(i for (i,) in self.connection.execute(
                    f"SELECT target FROM edges WHERE source IN ({marks})", chunk))
                found.update(i for (i,) in self.connection.execute(
                    f"SELECT source FROM edges WHERE target IN ({marks})", chunk))
            # Only concepts of an allowed sense are kept and walked on.
            allowed = set()
            for chunk in chunks(list(found - reached), queryChunk):
                marks = ",".join("?" * len(chunk))
                for i, neighbour, language, sense in self.connection.execute(
                        f"SELECT id, word, language, sense FROM concepts WHERE id IN ({marks})", chunk):
                    if sense.lower() not in self.blacklisted:
                        allowed.add(i)
                        if language == self.language:
                            words.add(neighbour)
            frontier = allowed
            reached |= found
        words.discard(word)
        self.cache[word] = sorted(words)
        return self.cache[word]

# ==============================================================================
# The spans of the terms whose words are replaced.
# ==============================================================================

def termSpans(terms : list, conceptnet : ConceptNet) -> dict:
    """
    Per term its spans (start, end, text in lower case): the runs of at most
    syngen.conceptnet.max_span tokens that start and end with a content word
    and are a ConceptNet concept, e.g. "short", "stature" and "short stature"
    in "Short stature". The tokens and parts of speech are those of the
    normalizer's spaCy model.
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
    Asks the LLM per (concept, span, term, neighbours) which neighbours can
    replace the span in the concept, syngen.conceptnet.filter_chunk
    neighbours per prompt, with syngen.conceptnet.filter_temperature. Returns
    the kept neighbours per (concept, span); a neighbour the answer does not
    name, or of an answer without a list, is dropped.
    """
    from model import Model

    prompts = []
    for hpoID, span, term, neighbours in questions:
        info = information[hpoID]
        for chunk in chunks(neighbours, conceptnetConfig.filter_chunk):
            prompts.append(((hpoID, span), chunk, {
                "hpo_label"   : quote(info["label"]),
                "hpo_details" : promptDetails(info, definition = bool(info["definition"]),
                    comment = False, structure = False),
                "term"        : quote(term),
                "token"       : quote(span),
                "candidates"  : "\n".join(f"* {neighbour}" for neighbour in chunk)
            }))
    l.log(f"{len(prompts)} filter prompts for {len(questions)} words.")

    kept = {}
    if len(prompts) == 0:
        return kept

    # The filter decides, so it samples (almost) greedily.
    configuration = configurationCopy()
    configuration["llm"]["temperature"] = conceptnetConfig.filter_temperature
    modelIndex = configuration["llm"]["model_id"].index(modelID)

    with l.step(f"Setting up the LLM ({modelNameOf(modelID)})"):
        model = Model(configuration, modelIndex)
    with model:
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

    if invalid > 0:
        l.warning(f"Filter: {invalid} of {len(prompts)} answers contain no list; their " \
            "neighbours are dropped.")
    l.log(f"Filter: {sum(len(v) for v in kept.values())} neighbours kept, {hallucinated} " \
        "kept words were no candidate.")
    return kept

# ==============================================================================
# Recombination: new terms with up to max_substitutions replaced spans.
# ==============================================================================

def substitute(term : str, replacements : tuple, choice : tuple) -> tuple[str, str]:
    """
    The term with the spans (start, end, text, neighbours) of `replacements`,
    which do not overlap and are in the order of the term, replaced by the
    words of `choice`, and how it was made, e.g. ("Small height", "short ->
    small; stature -> height").
    """
    new, position = "", 0
    for (start, end, _, _), neighbour in zip(replacements, choice):
        # ConceptNet's words are in lower case; a replaced word keeps the
        # capital letter of the original.
        if term[start].isupper() and neighbour[:1].islower():
            neighbour = neighbour[0].upper() + neighbour[1:]
        new += term[position:start] + neighbour
        position = end
    new += term[position:]
    return new, "; ".join(f"{r[2]} -> {n}" for r, n in zip(replacements, choice))

def recombine(term : str, replacements : list) -> list[tuple[str, str]]:
    """
    The new terms of a term and how they were made: the replacements of one
    span, taken from the spans in turn (so that the cap does not leave only
    the first span replaced), then of two non-overlapping spans, and so on up
    to syngen.conceptnet.max_substitutions, at most
    syngen.conceptnet.max_terms_per_seed. `replacements` holds per span
    (start, end, text, neighbours).
    """
    results = []
    perSpan = [[((r,), (n,)) for n in r[3]] for r in replacements]
    singles = (s for s in itertools.chain.from_iterable(itertools.zip_longest(*perSpan)) if s is not None)
    for count in range(1, conceptnetConfig.max_substitutions + 1):
        if count == 1:
            options = singles
        else:
            options = ((ordered, choice)
                for combination in itertools.combinations(replacements, count)
                for ordered in [tuple(sorted(combination, key = lambda r: r[0]))]
                if all(a[1] <= b[0] for a, b in zip(ordered, ordered[1:]))
                for choice in itertools.product(*[r[3] for r in ordered]))
        for ordered, choice in options:
            results.append(substitute(term, ordered, choice))
            if len(results) >= conceptnetConfig.max_terms_per_seed:
                return results
    return results

# ==============================================================================
# The steps of this run.
# ==============================================================================

def readJSON(file : str):
    """A JSON file of a step before, which must exist."""
    exitIfFileNotExist(file)
    with open(file, "r", encoding = "utf-8") as f:
        return json.load(f)

def findSpans(modelID : str) -> None:
    """
    The first step of the ConceptNet stage of the synonyms one LLM generated
    (normalizer): the seeds of the concepts, the spans of the seeds and per
    concept and span the first seed with it and the neighbours of the span.
    """
    modelName = modelNameOf(modelID)
    l.printHeader(f"Finding the Spans for ConceptNet ({modelName})")

    previous    = readSynonyms(outputFileStage(previousStage(conceptnetStage), modelName))
    information = conceptInformation(readTask())
    labels      = {hpoID : info["label"] for hpoID, info in information.items()}
    concepts    = [hpoID for hpoID in taskConcepts() if hpoID in labels]
    seeds       = seedTerms(previous, labels, concepts, conceptnetConfig.seed_sources,
        withLabel = "label" in conceptnetConfig.seed_sources)
    terms       = list(dict.fromkeys(term for t in seeds.values() for term in t))
    l.log(f"{len(terms)} terms of {len(seeds)} concepts, seed sources " \
        f"{applyFormat(conceptnetConfig.seed_sources)}.")

    conceptnet = ConceptNet(conceptnetConfig.database)
    try:
        with l.step("Finding the spans and their neighbours"):
            spans = termSpans(terms, conceptnet)
            neighbours = {text : conceptnet.neighbours(text)
                for found in spans.values() for _, _, text in found}
    finally:
        conceptnet.close()
    l.log(f"{sum(len(s) for s in spans.values())} spans, {len(neighbours)} distinct; " \
        f"{sum(len(n) for n in neighbours.values())} neighbours.")

    # Per concept and span: the first term with it (named in the filter
    # prompt) and the neighbours that may replace it.
    allowed = {}
    for hpoID, conceptTerms in seeds.items():
        for term in conceptTerms:
            for _, _, text in spans[term]:
                allowed.setdefault((hpoID, text), (term, neighbours[text]))

    writeJSON({
        "seeds"   : seeds,
        "spans"   : spans,
        "allowed" : [[hpoID, text, term, words] for (hpoID, text), (term, words) in allowed.items()]
    }, outputFileConceptNetSpans(modelName))

def filterSpans(modelID : str) -> None:
    """
    The second step of the ConceptNet stage (model): the LLM keeps the
    neighbours that can replace a span (filterNeighbours()).
    """
    modelName = modelNameOf(modelID)
    l.printHeader(f"Filtering the Neighbours ({modelName})")

    stored      = readJSON(outputFileConceptNetSpans(modelName))
    information = conceptInformation(readTask())
    with l.step("Filtering the neighbours with the LLM"):
        questions = [(hpoID, text, term, words)
            for hpoID, text, term, words in stored["allowed"] if len(words) > 0]
        kept = filterNeighbours(questions, information, modelID)
    writeJSON([[hpoID, text, sorted(words)] for (hpoID, text), words in kept.items()],
        outputFileConceptNetFilter(modelName))

def buildTerms(modelID : str) -> None:
    """
    The last step of the ConceptNet stage (normalizer): replaces the spans by
    their (kept) neighbours, ranks the new terms and writes the file of the
    stage.
    """
    modelName = modelNameOf(modelID)
    l.printHeader(f"Building Synonyms with ConceptNet ({modelName})")

    spansFile = outputFileConceptNetSpans(modelName)
    stored    = readJSON(spansFile)
    seeds     = stored["seeds"]
    spans     = {term : [tuple(span) for span in found] for term, found in stored["spans"].items()}
    allowed   = {(hpoID, text) : words for hpoID, text, _, words in stored["allowed"]}
    if conceptnetConfig.filter:
        filterFile = outputFileConceptNetFilter(modelName)
        if isOutdated(filterFile, [spansFile]):
            l.error(f"{quote(os.path.basename(filterFile))} is missing or older than " \
                f"{quote(os.path.basename(spansFile))}; run SynGenConceptNet.py --filter first.")
            sys.exit(1)
        kept    = {(hpoID, text) : words for hpoID, text, words in readJSON(filterFile)}
        allowed = {key : kept.get(key, []) for key in allowed}

    previous = readSynonyms(outputFileStage(previousStage(conceptnetStage), modelName))
    labels   = labelsByConcept(readTask())

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
    writeStage(conceptnetStage, modelName, previous, ranked, labels)

# The steps of the stage: option, action and function.
steps = {
    "--spans"  : ("Finding the spans for ConceptNet", findSpans),
    "--filter" : ("Filtering the neighbours with the LLM", filterSpans),
    "--build"  : ("Building synonyms with ConceptNet", buildTerms)
}

def main() -> None:
    """
    Runs one step of the ConceptNet stage of the experiment for every model.
    """
    startRun(allowedOptions = tuple(steps))
    chosen = [option for option in steps if option in options]
    if len(chosen) != 1:
        l.error(f"Give one of the steps {applyFormat(list(steps))}.")
        sys.exit(1)
    if conceptnetStage not in experimentStages:
        l.log(f"Experiment {experimentIndex} has no ConceptNet stage.")
        return
    if chosen[0] == "--filter" and not conceptnetConfig.filter:
        l.log("syngen.conceptnet.filter is off; the neighbours are not filtered.")
        return

    runForModels(*steps[chosen[0]])


# vLLM starts its workers with "spawn", which imports this script again: the
# steps must only run in the main process.
if __name__ == "__main__":
    main()
