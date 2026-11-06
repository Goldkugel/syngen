# ConceptNet

Written by `SynGenConceptNet.py <index>`: the synonyms of the stages before and those built with ConceptNet, one file per experiment and model (`conceptnet_<experiment>_LPC_<model>.csv`, or `_LC_` without the PubMed stage; columns as in [`../llm/README.md`](../llm/README.md)).

A ConceptNet synonym has its `seed` (the term whose words were replaced), `evidence` (the replacements, e.g. `short -> small`), `frequency` (from how many terms it was built) and its similarity to the label. Per concept the terms at least `syngen.conceptnet.min_similarity` similar to the label are kept, at most `syngen.conceptnet.max_candidates`.
