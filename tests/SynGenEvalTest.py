"""
Tests of the scores of SynGenEval.py on hand-made generated and gold
synonyms.
"""

import math

import numpy  as np
import pandas as pd

from SynGenEval import *

gold = pd.DataFrame({
    idColumn             : ["HP:1", "HP:1", "HP:2"],
    synonymColumn        : ["Small stature", "Low height", "Big head"],
    normalizedColumn     : ["small stature", "height low", "big head"],
    embeddingKeyColumn   : ["k1", "k2", "k3"],
    stageSimilarityColumn("llm")    : [1.0, 0.7, np.nan],
    stageSimilarityColumn("pubmed") : [np.nan, 1.0, 0.95],
})

generated = pd.DataFrame({
    idColumn             : ["HP:1", "HP:1", "HP:2", "HP:3"],
    synonymColumn        : ["small STATURE", "Height, low", "Large head", "Big head"],
    sourceColumn         : ["llm", "pubmed", "pubmed", "llm"],
    normalizedColumn     : ["small stature", "height low", "head large", "big head"],
    embeddingKeyColumn   : ["k1", "k2", "k4", "k3"],
    goldSimilarityColumn : [1.0, 1.0, 0.95, np.nan],
})

# As readEmbedded() adds it.
gold[lexicalKeyColumn]      = gold[synonymColumn].map(lexicalKey)
generated[lexicalKeyColumn] = generated[synonymColumn].map(lexicalKey)

def testLexicalCounts():
    scored = generated[generated[idColumn].isin(["HP:1", "HP:2"])]
    counts = conceptCounts(scored, gold, ["HP:1", "HP:2"], lexicalMatch, ["llm", "pubmed"])
    assert counts.loc["HP:1"].tolist() == [2, 1, 2, 1]
    assert counts.loc["HP:2"].tolist() == [1, 0, 1, 0]
    result = scores(counts)
    assert math.isclose(result["precision"], 1 / 3)
    assert math.isclose(result["recall"], 1 / 3)
    assert math.isclose(result["macro_recall"], 0.25)

def testNormalizedCounts():
    scored = generated[generated[idColumn].isin(["HP:1", "HP:2"])]
    counts = conceptCounts(scored, gold, ["HP:1", "HP:2"], normalizedMatch, ["llm", "pubmed"])
    assert counts["gold_matched"].sum() == 2 and counts["generated_matched"].sum() == 2

def testEmbeddingCountsPerView():
    llmOnly = generated[(generated[sourceColumn] == "llm") & generated[idColumn].isin(["HP:1", "HP:2"])]
    counts = conceptCounts(llmOnly, gold, ["HP:1", "HP:2"], embeddingMatch, ["llm"])
    assert counts["gold_matched"].sum() == 1 and counts["generated"].sum() == 1
    # At 0.9 the PubMed synonym "Large head" matches "Big head".
    scored = generated[generated[idColumn].isin(["HP:1", "HP:2"])]
    counts = conceptCounts(scored, gold, ["HP:1", "HP:2"], embeddingMatch, ["llm", "pubmed"], 0.9)
    assert counts["gold_matched"].sum() == 3 and counts["generated_matched"].sum() == 3

def testAccuracyIsConceptIndependent():
    # "Big head" is generated for HP:3, but is gold of HP:2.
    assert math.isclose(accuracy(generated, gold, lexicalMatch), 2 / 3)
    assert math.isclose(accuracy(generated, gold, embeddingMatch), 1.0)

def testWordsCovered():
    words = wordsCovered(generated[generated[idColumn] != "HP:3"], gold,
        {"HP:1" : {"short", "stature"}, "HP:2" : {"macrocephaly"}}, ["HP:1", "HP:2"])
    # HP:1: gold {small, stature, height, low} = generated; HP:2: {big, head}
    # against {head, large}.
    assert math.isclose(words["words_covered"], (1 + 0.5) / 2)
    assert math.isclose(words["words_jaccard"], (1 + 1 / 3) / 2)
    assert math.isclose(words["new_words_covered"], (1 + 0.5) / 2)

def testOtherMatches():
    other = pd.DataFrame({idColumn : ["HP:1", "HP:1"], synonymColumn : ["Dwarfism", "Tiny"],
        sourceColumn : ["related", "exact, narrow"], normalizedColumn : ["dwarfism", "tiny"]})
    labels = pd.DataFrame({idColumn : ["HP:2"], normalizedColumn : ["macrocephaly"]})
    found = pd.DataFrame({idColumn : ["HP:1"] * 4,
        normalizedColumn : ["dwarfism", "tiny", "big head", "macrocephaly"]})
    counts = otherMatches(found, other, gold, labels)
    assert counts == {"hpo_related" : 1, "hpo_broad" : 0, "hpo_narrow" : 0, "hpo_excluded" : 1,
        "other_concept" : 2}

def testBootstrapContainsEstimate():
    counts = pd.DataFrame({"gold" : [2, 1, 3], "gold_matched" : [1, 0, 3],
        "generated" : [2, 1, 4], "generated_matched" : [1, 0, 3]})
    intervals = bootstrap(counts, 200, 0)
    assert intervals["recall_low"] <= 4 / 6 <= intervals["recall_high"]
