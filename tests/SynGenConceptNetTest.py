"""
Tests of the ConceptNet stage on a database built from a tiny dump: the
concept URIs, the neighbours, the spans and the recombination.
"""

import gzip

import SynGenConceptNet
from SynGenConceptNetDB import build, parseConcept
from SynGenConceptNet   import *

assertions = [
    ("/r/Synonym", "/c/en/short/a", "/c/en/small/a"),
    ("/r/Synonym", "/c/en/small/a", "/c/en/little/a"),
    ("/r/Synonym", "/c/en/stature/n", "/c/en/height/n/wn/attribute"),
    ("/r/Synonym", "/c/en/short/a", "/c/en/brief/a/wn/food"),
    ("/r/FormOf", "/c/en/statures/n", "/c/en/stature/n"),
    ("/r/Synonym", "/c/en/short/a", "/c/de/kurz/a"),
    ("/r/RelatedTo", "/c/en/short/a", "/c/en/tall/a"),
]

def buildDatabase(tmp_path) -> str:
    dump = str(tmp_path / "dump.csv.gz")
    with gzip.open(dump, "wt", encoding = "utf-8") as f:
        for relation, source, target in assertions:
            f.write(f"/a/x\t{relation}\t{source}\t{target}\t{{\"weight\": 1.0}}\n")
    database = str(tmp_path / "conceptnet.sqlite")
    build(dump, database)
    return database

def testParseConcept():
    assert parseConcept("/c/en/short_stature/n") == ("short stature", "en", "n", "")
    assert parseConcept("/c/en/bank/n/wn/geography") == ("bank", "en", "n", "geography")
    assert parseConcept("/c/en/bank") == ("bank", "en", "", "")
    assert parseConcept("/r/Synonym") is None

def testNeighbours(tmp_path, monkeypatch):
    database = buildDatabase(tmp_path)
    conceptnet = ConceptNet(database)
    # Both directions, one hop, English only, without the blacklisted sense
    # "food" and the relation RelatedTo.
    assert conceptnet.neighbours("short") == ["small"]
    assert conceptnet.neighbours("stature") == ["height", "statures"]
    assert conceptnet.contains("small") and not conceptnet.contains("kurz")

    monkeypatch.setattr(SynGenConceptNet.conceptnetConfig, "hops", 2)
    conceptnet.cache.clear()
    assert conceptnet.neighbours("short") == ["little", "small"]
    conceptnet.close()

def testSpans(tmp_path):
    conceptnet = ConceptNet(buildDatabase(tmp_path))
    spans = termSpans(["Short stature"], conceptnet)["Short stature"]
    assert [text for _, _, text in spans] == ["short", "stature"]
    conceptnet.close()

def testRecombineCap(monkeypatch):
    # With a cap, every span is replaced before a span is replaced twice.
    monkeypatch.setattr(SynGenConceptNet.conceptnetConfig, "max_terms_per_seed", 2)
    terms = [new for new, _ in recombine("Short stature",
        [(0, 5, "short", ["small", "little", "tiny"]), (6, 13, "stature", ["height"])])]
    assert terms == ["Small stature", "Short height"]

def testRecombine():
    term = "Short stature"
    replacements = [(0, 5, "short", ["small", "Little"]), (6, 13, "stature", ["height"])]
    terms = [new for new, _ in recombine(term, replacements)]
    # One replacement from each span in turn, then two at once.
    assert terms == ["Small stature", "Short height", "Little stature", "Small height", "Little height"]
    # Overlapping spans are not replaced together.
    assert len(recombine("ab", [(0, 2, "ab", ["x"]), (1, 2, "b", ["y"])])) == 2

def testSeedTerms():
    previous = synonymFrame([
        {idColumn : "HP:1", synonymColumn : "Small stature", sourceColumn : "llm"},
        {idColumn : "HP:1", synonymColumn : "short stature", sourceColumn : "pubmed"},
        {idColumn : "HP:1", synonymColumn : "Stature", sourceColumn : "llm", asLabelColumn : True}])
    seeds = seedTerms(previous, {"HP:1" : "Short stature"}, ["HP:1"], ["llm"], withLabel = True)
    assert seeds == {"HP:1" : ["Short stature", "Small stature"]}
    seeds = seedTerms(previous, {"HP:1" : "Short stature"}, ["HP:1"], ["llm", "pubmed"], withLabel = False)
    assert seeds == {"HP:1" : ["Small stature", "short stature"]}
