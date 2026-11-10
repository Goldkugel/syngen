"""
The PubMed stage of syngen: searches the label and every LLM synonym of a
concept in the titles and abstracts of PubMed (up to syngen.pubmed.max_date),
finds biomedical terms in the sentences mentioning the searched term with a
NER model, and keeps per concept the terms most similar to its label. Writes
the synonyms of the stage before together with its own.

The search results and abstracts are kept in a SQLite cache, so that a term
is searched and an article fetched only once, and an interrupted run resumes.
On a cluster whose compute nodes have no internet access, the cache is filled
on the login node first (--fetch); the stage itself then runs offline.

    python3 ./src/SynGenPubMed.py <index> [<model>] --fetch   only searches and fetches
    python3 ./src/SynGenPubMed.py <index> [<model>]           the stage

The NCBI API key is read from the environment variable named by
syngen.pubmed.api_key_variable (never from the configuration file).
"""

import hashlib
import json
import os
import re
import sqlite3
import sys
import time
import xml.etree.ElementTree as ElementTree

import pandas as pd

sys.dont_write_bytecode = True

from SynGenUtils import *
from config      import *

from logger      import Logger

l = Logger(config = configFile)

pubmedConfig = syngenConfig.pubmed

eUtilitiesURL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
# PMIDs per efetch request; NCBI recommends at most 200 per GET, POST takes
# more.
fetchBatchSize = 200
# The earliest publication date of the search; esearch needs both dates.
minDate = "1000/01/01"
# Rows per SQL query with IN (SQLite allows 999 parameters in old versions).
queryChunk = 500
# The count of a search that failed; it is tried again by the next run.
failedSearch = -1

# The settings a search result depends on: a result made with others is
# searched again.
searchSettings = json.dumps({
    "field"    : pubmedConfig.field,
    "max_date" : pubmedConfig.max_date,
    "retmax"   : pubmedConfig.articles_per_query
}, sort_keys = True)

# ==============================================================================
# The cache: search results, articles and NER entities (SQLite).
# ==============================================================================

class PubMedCache:
    """
    The search results (PMIDs and total count per term and search settings;
    count failedSearch for a search that failed), the articles (title and
    abstract per PMID; empty for a PMID PubMed did not return) and the NER
    entities per sentence and NER model, in one SQLite file. Every write is
    committed, so an interrupted run loses nothing.
    """

    def __init__(self, path : str) -> None:
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok = True)
        # Several runs may use the cache at once; a writer waits for another.
        self.connection = sqlite3.connect(path, timeout = 600)
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS searches (
                term TEXT NOT NULL, settings TEXT NOT NULL, pmids TEXT NOT NULL,
                count INTEGER NOT NULL, fetched TEXT NOT NULL,
                PRIMARY KEY (term, settings));
            CREATE TABLE IF NOT EXISTS articles (
                pmid TEXT PRIMARY KEY, title TEXT NOT NULL, abstract TEXT NOT NULL,
                fetched TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS entities (
                sentence TEXT NOT NULL, model TEXT NOT NULL, spans TEXT NOT NULL,
                PRIMARY KEY (sentence, model));
        """)
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def searched(self, terms : list) -> dict:
        """
        The (PMIDs, total count) of the terms already searched with
        searchSettings, failed searches included.
        """
        found = {}
        for chunk in chunks(terms, queryChunk):
            rows = self.connection.execute(
                f"SELECT term, pmids, count FROM searches WHERE settings = ? AND term IN " \
                f"({','.join('?' * len(chunk))})", [searchSettings] + chunk)
            found.update({term : (json.loads(pmids), count) for term, pmids, count in rows})
        return found

    def addSearch(self, term : str, pmids : list, count : int) -> None:
        self.connection.execute("INSERT OR REPLACE INTO searches VALUES (?, ?, ?, ?, ?)",
            (term, searchSettings, json.dumps(pmids), count, time.strftime("%Y-%m-%d")))
        self.connection.commit()

    def articles(self, pmids : list) -> dict:
        """Title and abstract of the PMIDs already fetched."""
        found = {}
        for chunk in chunks(pmids, queryChunk):
            rows = self.connection.execute(
                f"SELECT pmid, title, abstract FROM articles WHERE pmid IN " \
                f"({','.join('?' * len(chunk))})", chunk)
            found.update({pmid : (title, abstract) for pmid, title, abstract in rows})
        return found

    def addArticles(self, articles : dict) -> None:
        today = time.strftime("%Y-%m-%d")
        self.connection.executemany("INSERT OR REPLACE INTO articles VALUES (?, ?, ?, ?)",
            [(pmid, title, abstract, today) for pmid, (title, abstract) in articles.items()])
        self.connection.commit()

    def entities(self, sentences : list, model : str) -> dict:
        """The entities [start, end, group] of the sentences already tagged with the model."""
        keys  = {sentenceKey(s) : s for s in sentences}
        found = {}
        for chunk in chunks(list(keys), queryChunk):
            rows = self.connection.execute(
                f"SELECT sentence, spans FROM entities WHERE model = ? AND sentence IN " \
                f"({','.join('?' * len(chunk))})", [model] + chunk)
            found.update({keys[h] : json.loads(spans) for h, spans in rows})
        return found

    def addEntities(self, entities : dict, model : str) -> None:
        self.connection.executemany("INSERT OR REPLACE INTO entities VALUES (?, ?, ?)",
            [(sentenceKey(s), model, json.dumps(v)) for s, v in entities.items()])
        self.connection.commit()

def sentenceKey(sentence : str) -> str:
    """The key of a sentence in the cache."""
    return hashlib.sha1(sentence.encode("utf-8")).hexdigest()

# ==============================================================================
# The E-utilities of PubMed: esearch (PMIDs of a term) and efetch (articles).
# ==============================================================================

class EUtilitiesError(RuntimeError):
    """A request to the E-utilities failed, or its answer could not be read."""

class EUtilities:
    """
    Requests to the E-utilities, at most syngen.pubmed.requests_per_second,
    retried with a growing pause after a failure that may pass (HTTP 429, 5xx,
    a connection error or timeout).
    """

    def __init__(self) -> None:
        import requests

        self.session    = requests.Session()
        self.interval   = 1.0 / pubmedConfig.requests_per_second
        self.last       = 0.0
        self.parameters = {"tool" : "syngen"}
        apiKey = os.environ.get(pubmedConfig.api_key_variable, "").strip()
        if apiKey:
            self.parameters["api_key"] = apiKey
        else:
            l.warning(f"No NCBI API key in ${pubmedConfig.api_key_variable}: at most 3 " \
                "requests per second are allowed.")
            self.interval = max(self.interval, 1.0 / 3)
        if pubmedConfig.email:
            self.parameters["email"] = pubmedConfig.email

    def request(self, utility : str, parameters : dict, post : bool = False) -> str:
        """
        The text of the answer of an E-utility.

        :raises EUtilitiesError: If it failed pubmed.retries + 1 times, or
            with an HTTP error that does not pass (4xx other than 429).
        """
        import requests

        parameters = self.parameters | parameters
        for attempt in range(pubmedConfig.retries + 1):
            pause = self.interval - (time.monotonic() - self.last)
            if pause > 0:
                time.sleep(pause)
            self.last = time.monotonic()
            try:
                if post:
                    response = self.session.post(eUtilitiesURL + utility, data = parameters, timeout = 120)
                else:
                    response = self.session.get(eUtilitiesURL + utility, params = parameters, timeout = 120)
                if response.status_code == 200:
                    return response.text
                reason = f"HTTP {response.status_code}"
                if 400 <= response.status_code < 500 and response.status_code != 429:
                    break
            except requests.RequestException as e:
                reason = type(e).__name__
            if attempt < pubmedConfig.retries:
                wait = 2 ** attempt
                l.warning(f"{utility} failed ({reason}); retrying in {wait} s.", cmdline = False)
                time.sleep(wait)
        raise EUtilitiesError(f"{utility} failed: {reason}.")

    def search(self, term : str) -> tuple[list, int]:
        """
        The PMIDs of the most relevant articles with the term (as a phrase)
        in syngen.pubmed.field, published up to max_date, and how many there
        are in total.

        :raises EUtilitiesError: If the request failed or its answer is no
            search result.
        """
        text = self.request("esearch.fcgi", {
            "db"       : "pubmed",
            "term"     : searchTerm(term),
            "retmax"   : pubmedConfig.articles_per_query,
            "sort"     : "relevance",
            "datetype" : "pdat",
            "mindate"  : minDate,
            "maxdate"  : pubmedConfig.max_date,
            "retmode"  : "json"
        })
        try:
            result = json.loads(text)["esearchresult"]
            return [str(pmid) for pmid in result.get("idlist", [])], int(result.get("count", 0))
        except (ValueError, KeyError, TypeError) as e:
            raise EUtilitiesError(f"esearch gave no search result ({type(e).__name__}).") from e

    def fetch(self, pmids : list) -> dict:
        """
        Title and abstract per PMID; ("", "") for one PubMed did not return.

        :raises EUtilitiesError: If the request failed or its answer is no
            XML.
        """
        text = self.request("efetch.fcgi", {
            "db" : "pubmed", "id" : ",".join(pmids), "retmode" : "xml"
        }, post = True)
        try:
            root = ElementTree.fromstring(text)
        except ElementTree.ParseError as e:
            raise EUtilitiesError(f"efetch gave no XML ({e}).") from e
        articles = {pmid : ("", "") for pmid in pmids}
        for article in root.iter("PubmedArticle"):
            pmid = article.findtext("MedlineCitation/PMID", default = "").strip()
            titleElement = article.find(".//ArticleTitle")
            title = "".join(titleElement.itertext()).strip() if titleElement is not None else ""
            # A structured abstract has several parts (background, methods, ...).
            abstract = " ".join("".join(part.itertext()).strip()
                for part in article.findall(".//Abstract/AbstractText"))
            if pmid in articles:
                articles[pmid] = (title, abstract)
        return articles

def searchTerm(term : str) -> str:
    """
    The esearch query of a term: the term as a phrase in the field, without
    the characters of PubMed's query syntax, e.g. "short stature"[tiab].
    """
    cleaned = re.sub(r"[\"\[\]()*:]", " ", term)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return f"\"{cleaned}\"[{pubmedConfig.field}]"

# ==============================================================================
# Searching the terms and fetching their articles.
# ==============================================================================

def fetchArticles(cache : PubMedCache, terms : list) -> None:
    """
    Searches every term not searched yet (or whose search failed) and fetches
    every article of the results not fetched yet. A failed search or fetch is
    logged and tried again by the next run; it does not stop the others.
    """
    terms   = list(dict.fromkeys(terms))
    done    = cache.searched(terms)
    missing = [term for term in terms if term not in done or done[term][1] == failedSearch]
    l.log(f"{len(terms)} terms, {len(terms) - len(missing)} already searched, " \
        f"{len(missing)} to search.")

    client = None
    if len(missing) > 0:
        client = EUtilities()
        started, failed = time.monotonic(), 0
        for index, term in enumerate(missing):
            try:
                pmids, count = client.search(term)
            except EUtilitiesError as e:
                l.warning(f"Searching {quote(term)} failed: {e}", cmdline = False)
                pmids, count = [], failedSearch
                failed += 1
            cache.addSearch(term, pmids, count)
            if (index + 1) % 1000 == 0 or index + 1 == len(missing):
                elapsed = time.monotonic() - started
                l.log(f"Searched {index + 1}/{len(missing)} terms, {failed} failed " \
                    f"(~{elapsed / (index + 1) * (len(missing) - index - 1) / 60:.0f} min left).")
        if failed > 0:
            l.warning(f"{failed} of {len(missing)} searches failed (see the log file); " \
                "they are tried again by the next run.")
        done = cache.searched(terms)

    pmids   = list(dict.fromkeys(pmid for term in terms for pmid in done.get(term, ([], 0))[0]))
    fetched = cache.articles(pmids)
    missing = [pmid for pmid in pmids if pmid not in fetched]
    l.log(f"{len(pmids)} articles, {len(fetched)} already fetched, {len(missing)} to fetch.")
    if len(missing) > 0:
        client = client or EUtilities()
        failed = 0
        batches = list(chunks(missing, fetchBatchSize))
        for index, batch in enumerate(batches):
            try:
                cache.addArticles(client.fetch(batch))
            except EUtilitiesError as e:
                l.warning(f"Fetching {len(batch)} articles (from PMID {batch[0]}) failed: {e}",
                    cmdline = False)
                failed += len(batch)
            if (index + 1) % 50 == 0 or index + 1 == len(batches):
                l.log(f"Fetched {min((index + 1) * fetchBatchSize, len(missing))}/{len(missing)} " \
                    f"articles, {failed} failed.")
        if failed > 0:
            l.warning(f"Fetching {failed} of {len(missing)} articles failed (see the log file); " \
                "they are tried again by the next run.")

# ==============================================================================
# Candidates: the NER entities of the sentences mentioning a searched term.
# ==============================================================================

# A sentence ends at ".", "!" or "?" followed by white space and an upper case
# letter, digit or bracket; "0.5" or "e.g. the" are not split.
_sentenceEnd = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")
_nonWord     = re.compile(r"\W+")

def sentencesOf(title : str, abstract : str) -> list[str]:
    """The title and the sentences of the abstract of an article."""
    parts = [title.strip()] + _sentenceEnd.split(abstract.strip())
    return [part.strip() for part in parts if len(part.strip()) > 0]

def wordKey(text : str) -> str:
    """
    The lexicalKey() of a text with every run of other characters than
    letters and digits as one space, and a space at both ends, so that a term
    is found in a sentence as whole words: " ear " is not in " year ".
    """
    return " " + _nonWord.sub(" ", lexicalKey(text)).strip() + " "

def plausible(candidate : str) -> bool:
    """
    Whether a NER span can be a synonym: at least syngen.pubmed.min_length
    characters, mostly letters, brackets closed, not cut off (a single
    character as last word) and no list (a comma); after the filters of the
    student project ba-synonym-generation.
    """
    candidate = candidate.strip()
    if len(candidate) < pubmedConfig.min_length:
        return False
    letters = sum(c.isalpha() for c in candidate)
    if letters < 2 or letters / len(candidate) < 0.5:
        return False
    if candidate.count("(") != candidate.count(")") or "[" in candidate or "]" in candidate:
        return False
    if "," in candidate or ";" in candidate:
        return False
    words = candidate.split()
    return len(words[-1]) > 1

def spansOf(entities : list) -> list[tuple[int, int]]:
    """
    The spans (start, end) of the entities [start, end, group] of a sentence
    that are of one of syngen.pubmed.entity_groups (all if it is empty), and
    of every two adjacent ones (at most 5 characters apart, e.g. "short" and
    "stature"), as NER often splits a term.
    """
    groups   = set(pubmedConfig.entity_groups)
    entities = sorted([e for e in entities if len(groups) == 0 or e[2] in groups], key = lambda e: e[0])
    spans    = [(e[0], e[1]) for e in entities]
    spans   += [(a[0], b[1]) for a, b in zip(entities, entities[1:]) if 0 <= b[0] - a[1] <= 5]
    return spans

class EntityTagger:
    """
    The NER model (syngen.pubmed.ner_model) on the GPU if there is one; its
    entities are cached per sentence.
    """

    def __init__(self, cache : PubMedCache) -> None:
        self.cache    = cache
        self.pipeline = None

    def load(self):
        """The NER pipeline, loaded on first use."""
        if self.pipeline is None:
            import torch
            from transformers import pipeline

            with l.step(f"Loading the NER model {quote(pubmedConfig.ner_model)}"):
                self.pipeline = pipeline("ner", model = pubmedConfig.ner_model,
                    aggregation_strategy = "simple", device = 0 if torch.cuda.is_available() else -1)
        return self.pipeline

    def entities(self, sentences : list) -> dict:
        """The entities [start, end, group] of every sentence."""
        sentences = list(dict.fromkeys(sentences))
        found   = self.cache.entities(sentences, pubmedConfig.ner_model)
        missing = [s for s in sentences if s not in found]
        l.log(f"{len(sentences)} sentences, {len(found)} already tagged, {len(missing)} to tag.")
        for batch in chunks(missing, 10 * pubmedConfig.ner_batch_size):
            results = self.load()(batch, batch_size = pubmedConfig.ner_batch_size)
            # A single sentence may come back without the outer list.
            if len(batch) == 1 and (len(results) == 0 or isinstance(results[0], dict)):
                results = [results]
            tagged = {sentence : [[int(e["start"]), int(e["end"]), str(e.get("entity_group", ""))]
                for e in entities] for sentence, entities in zip(batch, results)}
            self.cache.addEntities(tagged, pubmedConfig.ner_model)
            found.update(tagged)
        return found

def findCandidates(cache : PubMedCache, seeds : dict) -> pd.DataFrame:
    """
    The candidates of every concept: the plausible NER spans of the sentences
    of the articles found for its terms (with syngen.pubmed.sentences "seed"
    only of those mentioning the searched term), one row per concept and
    candidate (lexicalKey()) with the term that found it first, its first
    PMIDs and how often it was found.
    """
    allTerms   = list(dict.fromkeys(term for terms in seeds.values() for term in terms))
    searched   = cache.searched(allTerms)
    unsearched = [term for term in allTerms if term not in searched]
    if len(unsearched) > 0:
        raise RuntimeError(f"{len(unsearched)} terms have not been searched; run with --fetch " \
            "(e.g. on a node with internet access) first.")
    failed = [term for term in allTerms if searched[term][1] == failedSearch]
    if len(failed) > 0:
        l.warning(f"The search of {len(failed)} of {len(allTerms)} terms failed; they have no " \
            "articles. Run with --fetch to search them again.")

    pmids    = list(dict.fromkeys(p for term in allTerms for p in searched[term][0]))
    articles = cache.articles(pmids)
    if len(articles) < len(pmids):
        l.warning(f"{len(pmids) - len(articles)} of {len(pmids)} articles have not been fetched; " \
            "run with --fetch to fetch them.")
    # Every sentence with its key, to find the term in it.
    sentences = {pmid : [(s, wordKey(s)) for s in sentencesOf(*articles.get(pmid, ("", "")))]
        for pmid in pmids}

    # The sentences of every (concept, term, PMID) to tag.
    selected = []
    for hpoID, terms in seeds.items():
        for term in terms:
            key = wordKey(term)
            for pmid in searched[term][0]:
                for sentence, sentenceWords in sentences[pmid]:
                    if pubmedConfig.sentences == "all" or key in sentenceWords:
                        selected.append((hpoID, term, pmid, sentence))
    l.log(f"{len(selected)} sentences of {len(pmids)} articles to search for candidates " \
        f"(syngen.pubmed.sentences {quote(pubmedConfig.sentences)}).")

    with l.step("Finding the entities of the sentences"):
        entities = EntityTagger(cache).entities([sentence for _, _, _, sentence in selected])

    found = {}
    for hpoID, term, pmid, sentence in selected:
        for start, end in spansOf(entities.get(sentence, [])):
            candidate = sentence[start:end].strip()
            if not plausible(candidate):
                continue
            entry = found.setdefault((hpoID, lexicalKey(candidate)), {
                idColumn : hpoID, synonymColumn : candidate, sourceColumn : pubmedStage,
                seedColumn : term, "pmids" : [], frequencyColumn : 0})
            entry[frequencyColumn] += 1
            if pmid not in entry["pmids"]:
                entry["pmids"].append(pmid)

    for entry in found.values():
        entry[evidenceColumn] = " ".join(entry.pop("pmids")[:5])
    candidates = synonymFrame(list(found.values()))
    l.log(f"{len(candidates.index)} candidates of {candidates[idColumn].nunique()} concepts.")
    return candidates

# ==============================================================================
# The steps of this run.
# ==============================================================================

def runModel(modelID : str, fetchOnly : bool) -> None:
    """
    The PubMed stage of the synonyms one LLM generated: searches and fetches,
    and unless fetchOnly finds, ranks and writes the candidates.
    """
    modelName = modelNameOf(modelID)
    l.printHeader(f"Finding Synonyms in PubMed ({modelName})")

    labels   = labelsByConcept(readTask())
    previous = readPreviousStage(pubmedStage, modelName, labels)
    concepts = [hpoID for hpoID in taskConcepts() if hpoID in labels]
    seeds    = seedTerms(previous, labels, concepts, [llmStage], withLabel = True)
    l.log(f"{sum(len(t) for t in seeds.values())} terms of {len(seeds)} concepts to search, " \
        f"articles up to {pubmedConfig.max_date}.")

    cache = PubMedCache(pubmedConfig.cache_file)
    try:
        with l.step("Searching the terms and fetching the articles"):
            fetchArticles(cache, [term for terms in seeds.values() for term in terms])
        if fetchOnly:
            return
        candidates = findCandidates(cache, seeds)
    finally:
        cache.close()

    with l.step("Ranking the candidates"):
        found = rankCandidates(candidates, previous, labels,
            pubmedConfig.min_similarity, pubmedConfig.max_candidates)
    writeStage(pubmedStage, modelName, previous, found, labels)

def main() -> None:
    """
    Runs the PubMed stage of the experiment for every model.
    """
    startRun(allowedOptions = ("--fetch",))
    if pubmedStage not in experimentStages:
        l.log(f"Experiment {experimentIndex} has no PubMed stage.")
        return

    fetchOnly = "--fetch" in options
    runForModels("Fetching the PubMed articles" if fetchOnly else "Finding synonyms in PubMed",
        lambda modelID: runModel(modelID, fetchOnly))


if __name__ == "__main__":
    main()
