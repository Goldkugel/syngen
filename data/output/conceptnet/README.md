# ConceptNet

Written by `SynGenConceptNet.py <index> --build`: the synonyms of the stages before and those built with ConceptNet, one file per experiment and model (`conceptnet_<experiment>_LPC_<model>.csv`, or `_LC_` without the PubMed stage; columns as in [`../llm/README.md`](../llm/README.md)).

A ConceptNet synonym has its `seed` (the term whose words were replaced), `evidence` (the replacements, e.g. `short -> small`), `frequency` (from how many terms it was built) and its similarity to the label. Per concept the terms at least `syngen.conceptnet.min_similarity` similar to the label are kept, at most `syngen.conceptnet.max_candidates`.

The steps before it hand their results on in two files, as the spans need the normalizer and the filter the LLM, which run in different Python environments:

| File | Content |
|---|---|
| `conceptnet_<experiment>_LPC_spans_<model>.json` | `--spans`: the seeds per concept, the spans per seed (start, end, text) and per concept and span the first seed with it and the neighbours of the span |
| `conceptnet_<experiment>_LPC_filter_<model>.json` | `--filter`: the neighbours the LLM kept per concept and span (only with `syngen.conceptnet.filter`) |
