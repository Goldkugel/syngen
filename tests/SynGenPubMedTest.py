"""
Tests of the PubMed stage without network: the query, the sentences, the
candidate filter, the parsing of efetch and the cache.
"""

import SynGenPubMed
from SynGenPubMed import *


def testSearchTerm():
    assert searchTerm("Short stature") == "\"Short stature\"[tiab]"
    assert searchTerm("Abnormality (of) the \"eye\" [x]") == "\"Abnormality of the eye x\"[tiab]"

def testSentencesOf():
    sentences = sentencesOf("A title", "First sentence with 0.5 mg. Second one, e.g. this. (Third) here!")
    assert sentences == ["A title", "First sentence with 0.5 mg.", "Second one, e.g. this.", "(Third) here!"]

def testPlausible():
    assert plausible("short stature")
    assert not plausible("ab")
    assert not plausible("12.5 mg/kg")
    assert not plausible("growth (delay")
    assert not plausible("Ki67, ZEB")
    assert not plausible("ICA o")

def testFetchParsesTitlesAndAbstracts(monkeypatch):
    answer = """<?xml version="1.0"?>
    <PubmedArticleSet>
      <PubmedArticle><MedlineCitation><PMID>1</PMID><Article>
        <ArticleTitle>Short <i>stature</i></ArticleTitle>
        <Abstract><AbstractText Label="A">First.</AbstractText><AbstractText>Second.</AbstractText></Abstract>
      </Article></MedlineCitation></PubmedArticle>
    </PubmedArticleSet>"""
    client = EUtilities()
    monkeypatch.setattr(client, "request", lambda utility, parameters, post = False: answer)
    articles = client.fetch(["1", "2"])
    assert articles == {"1" : ("Short stature", "First. Second."), "2" : ("", "")}

def testSearchRequest(monkeypatch):
    sent = {}
    def request(utility, parameters, post = False):
        sent.update(parameters)
        return '{"esearchresult": {"count": "42", "idlist": ["3", "1"]}}'
    client = EUtilities()
    monkeypatch.setattr(client, "request", request)
    assert client.search("Short stature") == (["3", "1"], 42)
    assert sent["maxdate"] == syngenConfig.pubmed.max_date
    assert sent["datetype"] == "pdat"

def testCache(tmp_path):
    cache = PubMedCache(str(tmp_path / "cache.sqlite"))
    cache.addSearch("a", ["1", "2"], 2)
    cache.addArticles({"1" : ("T", "A"), "2" : ("", "")})
    cache.addSpans({"A sentence." : [[0, 1]]}, "ner")
    assert cache.searched(["a", "b"]) == {"a" : ["1", "2"]}
    assert cache.articles(["1", "3"]) == {"1" : ("T", "A")}
    assert cache.spans(["A sentence.", "Other."], "ner") == {"A sentence." : [[0, 1]]}
    assert cache.spans(["A sentence."], "other model") == {}
    cache.close()

def testFindCandidates(tmp_path, monkeypatch):
    cache = PubMedCache(str(tmp_path / "cache.sqlite"))
    cache.addSearch("Short stature", ["1"], 1)
    sentence = "Short stature, also called small stature, is common."
    cache.addArticles({"1" : ("", sentence + " Unrelated growth hormone sentence.")})
    start = sentence.index("small")
    # Two adjacent entities are joined as well.
    cache.addSpans({sentence : [[start, start + 5], [start + 6, start + 13], [start, start + 13]]},
        syngenConfig.pubmed.ner_model)
    candidates = findCandidates(cache, {"HP:1" : ["Short stature"]})
    assert sorted(candidates[synonymColumn]) == ["small", "small stature", "stature"]
    assert set(candidates[evidenceColumn]) == {"1"}
    cache.close()
