# LLM

Written by `SynGenLLM.py <index>`, one pair of files per experiment and model (e.g. `llm_Test_Def_Com_Str_FS_R5_L_<model>.csv`).

| File | Content |
|---|---|
| `llm_<experiment>_raw_<model>.csv` | The answer of every concept and round: `id`, `round`, `answer`, `valid` (whether it holds a list of synonyms) and `synonyms` (the parsed list, as JSON) |
| `llm_<experiment>_L_<model>.csv` | The synonyms: one row per concept and synonym (case and white space ignored), in its most frequent spelling, with `rounds` (in how many rounds it was generated) and `first_round`, and `as_label` (equal to the label lexically, case and white space ignored; the normalizer runs in another Python environment than the LLM, so the stages after it and `SynGenEmbed.py` add the normalised match) |

The columns of all synonym files are `id`, `synonym`, `source` (the stage), `model`, `rounds`, `first_round`, `seed`, `evidence`, `frequency`, `similarity_<embedding model>` (to the label) and `as_label`; a stage leaves the columns of the others empty.
