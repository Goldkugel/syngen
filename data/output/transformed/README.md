# Transformed

Written by the preparation run (`python3 ./src/SynGenLLM.py`) and by `SynGenDevelopmentSet.py`, in the format of the `adapter` package: one row per concept (`id`), attribute (`label`, `definition`, `comment`, `synonym`, `child`) and value, with a dict of `additional` information (e.g. `{"semantic_class": "exact", "source_type": "layperson"}` for a synonym). Each file has a `<file>.settings.json` with the settings it was made with; it is made again when they change or hp.owl is newer.

| File | Content |
|---|---|
| `hpo.csv` | `hp.owl` transformed, without obsolete concepts, with all synonyms; the gold standard is taken from it (`goldStandard()` in `src/SynGenUtils.py`): the synonyms listed only as exact, not of a type of `syngen.gold.exclude_types` and not equal to their label |
| `hpo.development.csv` | The development set: concepts with gold synonyms drawn at random until their gold synonyms number `syngen.development_set.synonym_count`, with their label, definition, comment, synonyms and child relations in both directions, and the labels of their parents and children. Runs with `test` generate for them |
| `hpo.evaluation.csv` | `hpo.csv` without the synonyms of the development concepts (their labels stay, as parents and children of other concepts). Runs without `test` generate for all other concepts |
| `hpo.fewshot.json` | The few-shot examples: `syngen.development_set.few_shot_count` development concepts with at least two gold synonyms, with their information and gold synonyms. They are scored in no run |
