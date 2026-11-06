# Prompts

The prompt templates, in the format of the `model` package (`Model.addPromptsFromFile()`): `{placeholder}` fields, with literal braces written as `{{` and `}}`. They are listed in `llm.prompt_files` of [`config/config.yaml`](../config/config.yaml) and used by their index.

| Index | File | Used by | Answer |
|---|---|---|---|
| 0 | `generation.md` | `SynGenLLM.py`: the exact synonyms of a concept | `{"synonyms": [...]}` |
| 1 | `conceptnet_filter.md` | `SynGenConceptNet.py`: the ConceptNet neighbours that can replace a word of a term | `{"keep": [...]}` |

The `model` package records the hashes of the templates in the run metadata. A template can be edited in place: the prompts made from it are new, so their answers are generated anew, while the answers to unchanged prompts are taken from the answer store.

The templates name no real HPO synonyms, so that no gold synonym of an evaluated concept is in a prompt; examples come only from the few-shot examples, which are drawn from the development set and scored in no run.

## Placeholders

| Placeholder | Template | Value |
|---|---|---|
| `{hpo_label}` | both | The label of the concept, quoted |
| `{hpo_details}` | both | One line per information the experiment adds, each starting with a line break: `* Definition: ...` (`definition`), `* Comment: ...` (`comment`), `* Parents: ...` and `* Children: ...` (`structure`); for the filter only the definition; empty without them |
| `{examples}` | `generation.md` | Two line breaks and the few-shot examples (`hpo.fewshot.json`) in the input and output format of the prompt, for experiments with `few_shot`; otherwise empty |
| `{max_synonyms}` | `generation.md` | `syngen.llm.max_synonyms` |
| `{term}` | `conceptnet_filter.md` | The term whose word is replaced (the label or a synonym) |
| `{token}` | `conceptnet_filter.md` | The word or phrase replaced |
| `{candidates}` | `conceptnet_filter.md` | The ConceptNet neighbours, one per line, at most `syngen.conceptnet.filter_chunk` |
