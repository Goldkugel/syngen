# PubMed

Written by `SynGenPubMed.py <index>`: the synonyms of the stage before (the LLM) and those found in PubMed, one file per experiment and model (`pubmed_<experiment>_LP_<model>.csv`, columns as in [`../llm/README.md`](../llm/README.md)).

A PubMed synonym has its `seed` (the searched term: the label or an LLM synonym), `evidence` (the first five PMIDs it was found in), `frequency` (how often it was found) and its similarity to the label. Per concept the candidates at least `syngen.pubmed.min_similarity` similar to the label are kept, at most `syngen.pubmed.max_candidates`.

The search results, articles and NER spans are cached in `syngen.pubmed.cache_file` (`data/cache/pubmed/pubmed.sqlite`), which `cleanUp.sh` keeps.
