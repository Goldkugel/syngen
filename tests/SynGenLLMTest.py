"""
Tests of the LLM stage without a model: the prompt values and the counting of
the rounds.
"""

import pandas as pd

from SynGenLLM import *


def testCountRounds():
    found = pd.DataFrame([
        ("HP:1", 1, "Small stature", "small stature"),
        ("HP:1", 2, "small stature", "small stature"),
        ("HP:1", 3, "small stature", "small stature"),
        ("HP:1", 3, "Dwarfism", "dwarfism"),
        ("HP:2", 2, "Big head", "big head"),
    ], columns = [idColumn, "round", synonymColumn, "key"])
    synonyms = countRounds(found)
    rows = {(r[idColumn], r[synonymColumn]) : (r[roundsColumn], r[firstRoundColumn])
        for _, r in synonyms.iterrows()}
    # The most frequent spelling, the rounds and the first round.
    assert rows == {("HP:1", "small stature") : (3, 1), ("HP:1", "Dwarfism") : (1, 3),
        ("HP:2", "Big head") : (1, 2)}
    assert set(synonyms[sourceColumn]) == {llmStage}

def testPromptDetails():
    information = {"label" : "Seizure", "definition" : "A fit.", "comment" : "",
        "parents" : ["Abnormal nervous system physiology"], "children" : []}
    assert promptDetails(information, True, False, True) == \
        "\n* Definition: \"A fit.\"\n* Parents: \"Abnormal nervous system physiology\"\n* Children: None"
    assert promptDetails(information, False, False, False) == ""
    values = promptValues(information, "")
    assert values["hpo_label"] == "\"Seizure\"" and values["max_synonyms"] == syngenConfig.llm.max_synonyms

def testRunForModelsContinuesAfterAFailure(monkeypatch):
    import SynGenUtils
    monkeypatch.setattr(SynGenUtils, "modelIDs", ["org/a", "org/b"])
    called = []
    def function(modelID):
        called.append(modelID)
        if modelID == "org/a":
            raise ValueError("broken")
    try:
        SynGenUtils.runForModels("Testing", function)
        assert False, "runForModels() must end the run when a model failed"
    except SystemExit as e:
        assert e.code == 1
    assert called == ["org/a", "org/b"]
