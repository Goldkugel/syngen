# Embedded

Written by `SynGenEmbed.py <index>`, one pair of files per experiment and model, each with a `<file>.settings.json`.

| File | Content |
|---|---|
| `embed_<experiment>_<stages>_<model>.csv` | The synonyms of all stages (the file of the last stage) with `normalized` (the sorted tokens of the normalizer), `embedding_key` (equal for synonyms of the same embedding, i.e. cosine similarity 1 up to rounding), the similarity to the label and `gold_similarity` (the highest similarity to a gold synonym of the concept) |
| `embed_<experiment>_<stages>_<model>_gold.csv` | The gold synonyms of the concepts of the experiment with `normalized`, `embedding_key` and `similarity_<stage>` (the highest similarity to a synonym of the stage, as_label synonyms left out) |
