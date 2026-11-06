# Logs

Written by every script through the `logger` package (`logger` in [`config/config.yaml`](../../config/config.yaml)) and by the `model` package (`llm.prompt_log_*`, `llm.metadata_*`).

| File | Content |
|---|---|
| `syngen_<start>.log` | The log of one run, named after its start to the microsecond: the command line as its first row, then every step with its start, end and duration |
| `prompts_<model>_<experiment>_round<r>.log` | Every prompt and answer of a round of the LLM stage (`model.logPrompts()`) |
| `prompts_<model>_<experiment>_<stages>_filter.log` | Every prompt and answer of the ConceptNet filter |
| `run_metadata.json` | The settings of the runs of the `model` package: model revisions, sampling settings and the hashes of the prompt templates |

`cleanUp.sh` deletes the `.log` files.
