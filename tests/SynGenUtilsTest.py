"""
Tests of the helpers of SynGenUtils.py and of the configuration schema.
"""

import pandas as pd
import pytest

from pydantic     import ValidationError

from SynGenConfig import ExperimentConfig
from SynGenUtils  import *


def transformed(rows : list) -> pd.DataFrame:
    """A transformed HPO from (id, attribute, value, additional) tuples."""
    return pd.DataFrame(rows, columns = transformedColumns)

def synonym(hpoID : str, value : str, classes : str = "exact", types : str = "") -> tuple:
    additional = {"semantic_class" : classes}
    if types:
        additional["source_type"] = types
    return (hpoID, "synonym", value, additional)

hpo = transformed([
    ("HP:1", "label", "Short stature", {}),
    ("HP:1", "definition", "A height below the norm.", {}),
    ("HP:1", "child", "HP:2", {}),
    synonym("HP:1", "Small stature"),
    synonym("HP:1", "Decreased body height", "exact, related"),
    synonym("HP:1", "short  STATURE"),
    synonym("HP:1", "Dwarfism", "related"),
    synonym("HP:1", "Height loss", "exact", "obsolete"),
    synonym("HP:1", "small stature"),
    ("HP:2", "label", "Proportionate short stature", {}),
    synonym("HP:2", "Proportionate dwarfism"),
])


def testLexicalKey():
    assert lexicalKey("  Short\tStature ") == "short stature"
    assert lexicalKey("ＡＢＣ") == "abc"

def testGoldStandard():
    gold = goldStandard(hpo)
    # Only exact, not obsolete, not the label, each once.
    assert sorted(zip(gold[idColumn], gold[synonymColumn])) == [
        ("HP:1", "Small stature"), ("HP:2", "Proportionate dwarfism")]

def testOtherSynonyms():
    other = otherSynonyms(hpo)
    classes = dict(zip(other[synonymColumn], other[sourceColumn]))
    assert classes["Dwarfism"] == "related"
    assert classes["Decreased body height"] == "exact, related"
    assert "Small stature" not in classes

def testConceptInformation():
    information = conceptInformation(hpo)
    assert information["HP:1"]["children"] == ["Proportionate short stature"]
    assert information["HP:2"]["parents"] == ["Short stature"]
    assert information["HP:1"]["definition"] == "A height below the norm."
    assert information["HP:2"]["definition"] == ""

def testParseTerms():
    assert parseTerms({"synonyms" : [" a ", "", 3, None, True]}, "synonyms") == ["a", "3"]
    assert parseTerms(["x"], "synonyms") == ["x"]
    assert parseTerms({"other" : []}, "synonyms") is None
    assert parseTerms(None, "synonyms") is None

def testApplyFormat():
    assert applyFormat([]) == "None"
    assert applyFormat(["a", "b"]) == "\"a\", \"b\""

def testNormalizedKeysIgnoreOrderAndStopWords():
    keys = normalizedKeys(["Hypoplasia of the radius", "radius hypoplasia", "the"])
    assert keys[0] == keys[1] != ""
    assert keys[2] == ""

def testMarkAsLabel():
    synonyms = synonymFrame([
        {idColumn : "HP:1", synonymColumn : "SHORT stature"},
        {idColumn : "HP:1", synonymColumn : "Stature, short"},
        {idColumn : "HP:1", synonymColumn : "Small stature"}])
    marked = markAsLabel(synonyms, {"HP:1" : "Short stature"})
    assert marked[asLabelColumn].tolist() == [True, True, False]

def testLexicalMarkingIsCompletedByTheNextStage(tmp_path, monkeypatch):
    # The LLM stage marks without the normalizer, the next stage adds the
    # normalised match.
    import SynGenUtils
    file = str(tmp_path / "llm.csv")
    monkeypatch.setattr(SynGenUtils, "outputFileStage", lambda stage, modelName: file)
    synonyms = synonymFrame([
        {idColumn : "HP:1", synonymColumn : "SHORT stature", sourceColumn : llmStage},
        {idColumn : "HP:1", synonymColumn : "Stature, short", sourceColumn : llmStage}])
    labels = {"HP:1" : "Short stature"}
    marked = markAsLabel(synonyms, labels, normalized = False)
    assert marked[asLabelColumn].tolist() == [True, False]

    writeHugeCSV(marked, file)
    assert readPreviousStage(pubmedStage, "model", labels)[asLabelColumn].tolist() == [True, True]

def testStageWithoutCandidatesKeepsAsLabel(tmp_path, monkeypatch):
    # A stage that finds nothing keeps the as_label of the stages before.
    import SynGenUtils
    file = str(tmp_path / "stage.csv")
    monkeypatch.setattr(SynGenUtils, "outputFileStage", lambda stage, modelName: file)
    previous = markAsLabel(synonymFrame([
        {idColumn : "HP:1", synonymColumn : "short stature", sourceColumn : llmStage},
        {idColumn : "HP:1", synonymColumn : "Small stature", sourceColumn : llmStage}]),
        {"HP:1" : "Short stature"})
    writeStage(pubmedStage, "model", previous, synonymFrame([]), {"HP:1" : "Short stature"})
    assert readSynonyms(file)[asLabelColumn].tolist() == [True, False]

def testRankCandidates(monkeypatch):
    import SynGenUtils
    monkeypatch.setattr(SynGenUtils, "similarities",
        lambda texts, references: [{"a" : 0.9, "b" : 0.95, "c" : 0.5, "d" : 0.99}[t] for t in texts])
    candidates = synonymFrame([{idColumn : "HP:1", synonymColumn : s} for s in ["a", "b", "c", "d", "Label"]])
    previous   = synonymFrame([{idColumn : "HP:1", synonymColumn : "D"}])
    ranked = SynGenUtils.rankCandidates(candidates, previous, {"HP:1" : "label"}, 0.8, 1)
    # d is known, c not similar, Label the label; b the most similar.
    assert ranked[synonymColumn].tolist() == ["b"]

def testRankCandidatesComparesNormalised(monkeypatch):
    import SynGenUtils
    similarity = {"Short statures" : 0.99, "Stature, short" : 0.98, "Small stature" : 0.95,
        "small statures" : 0.97, "Height deficit" : 0.96, "the" : 0.9}
    monkeypatch.setattr(SynGenUtils, "similarities",
        lambda texts, references: [similarity[t] for t in texts])
    candidates = synonymFrame([{idColumn : "HP:1", synonymColumn : s} for s in similarity])
    previous   = synonymFrame([{idColumn : "HP:1", synonymColumn : "Deficit of height"}])
    ranked = SynGenUtils.rankCandidates(candidates, previous, {"HP:1" : "Short stature"}, 0.8, 10)
    # The label inflected or reordered, and a known synonym reordered, are
    # dropped; of "Small stature" and "small statures" the more similar is
    # kept; "the" has no normalised key and is compared lexically only.
    assert ranked[synonymColumn].tolist() == ["small statures", "the"]
    assert "_key" not in ranked.columns

def testExperimentStages():
    assert ExperimentConfig().stages == ["llm", "pubmed", "conceptnet"]
    assert ExperimentConfig(stages = ["llm", "conceptnet"]).stages == ["llm", "conceptnet"]
    with pytest.raises(ValidationError):
        ExperimentConfig(stages = ["pubmed"])
    with pytest.raises(ValidationError):
        ExperimentConfig(stages = ["llm", "conceptnet", "pubmed"])
    with pytest.raises(ValidationError):
        ExperimentConfig(stages = ["llm", "llm"])
