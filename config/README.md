# Configuration

[`config.yaml`](config.yaml) holds the settings of syngen and of the packages it uses, one section each. Paths are relative to the repository root, from where the scripts are run.

| Section | Read by | Content |
|---|---|---|
| `syngen` | `src/config.py` (validated by `src/SynGenConfig.py`) | The experiments and syngen's own settings (below) |
| `adapter` | the `adapter` package (`HPOAdapter`) | Where `hp.owl` is, and where the transformed HPO goes; the development and evaluation set are named after it in the same folder |
| `logger` | the `logger` package | Log folder and file name (one file per run), format and runtime display |
| `llm` | the `model` package | The model (`model_id`), sampling and vLLM settings, the prompt templates and the answer store and prompt log. `seed` is the seed of the first round of the LLM stage; `temperature` (0.95) is high so that the rounds differ; the ConceptNet filter uses `syngen.conceptnet.filter_temperature` instead |
| `embedder` | the `embedder` package | The embedding model of the similarities (SapBERT; only the first of `model_id` is used) with its pinned revision, pooling, device and cache |
| `normalizer` | the `normalizer` package | The normalisation of the normalised match and of the words covered |

The settings of the package sections are described in the READMEs of the packages. A key a schema does not know is refused, so typos are found when a run starts; so are unknown command-line options, more than two arguments and a model ID that is not in `llm.model_id`.

Some settings are constants in the code, as they rarely need a change: the earliest publication date of the PubMed search (`minDate`), the PMIDs per efetch request (`fetchBatchSize`), the most characters between two NER entities joined into one candidate (5, `spansOf()`), the decimals of the embedding key (`embeddingKeyDecimals`) and the Slurm resources (`#SBATCH` lines of `generate.sh`).

## `syngen`

| Key | Type | Default | Description |
|---|---|---|---|
| `experiments` | list | (required) | The experiments, picked by their index (from 0) on the command line; see below |
| `development_set.synonym_count` | `int` | `500` | Concepts with gold synonyms are drawn until their gold synonyms number at least this (the last concept is taken whole) |
| `development_set.seed` | `int` | `0` | Seed of the random selection |
| `development_set.few_shot_count` | `int` | `3` | Development concepts with at least two gold synonyms used as few-shot examples; scored in no run |
| `data_folder` | `str` | `"./data"` | Folder of the data |
| `gold.exclude_types` | list | `["obsolete"]` | Synonym types (`source_type` of the adapter) left out of the gold standard, besides synonyms not listed only as exact and synonyms equal to their label |
| `llm.rounds` | `int` | `5` | How often every concept is prompted, round r with seed `llm.seed + r - 1` |
| `llm.max_synonyms` | `int` | `10` | The most synonyms the prompt asks for |
| `pubmed.max_date` | `str` | `"2026/06/30"` | Only articles published up to this date (`YYYY/MM/DD`), so that a search finds the same articles again |
| `pubmed.articles_per_query` | `int` | `20` | Articles per searched term, the most relevant first |
| `pubmed.field` | `str` | `"tiab"` | The PubMed field searched (title and abstract) |
| `pubmed.api_key_variable` | `str` | `"NCBI_API_KEY"` | The environment variable holding the NCBI API key |
| `pubmed.requests_per_second` | `float` | `3` | At most 3 without and 10 with an API key |
| `pubmed.retries` | `int` | `5` | Retries of a failed request, with a growing pause |
| `pubmed.email` | `str` or `null` | `null` | Sent to NCBI with every request, as their guidelines ask |
| `pubmed.cache_file` | `str` | `"./data/cache/pubmed/pubmed.sqlite"` | The cache of searches, articles and NER spans |
| `pubmed.ner_model` | `str` | `"d4data/biomedical-ner-all"` | The NER model finding the candidates |
| `pubmed.ner_batch_size` | `int` | `64` | Sentences per NER batch |
| `pubmed.entity_groups` | list | `[]` | The entity groups of the NER model that are candidates (e.g. `Sign_symptom`, `Disease_disorder`); all if empty. The cache keeps the entities of all groups, so changing it needs no new tagging |
| `pubmed.sentences` | `"seed"` or `"all"` | `"seed"` | Only the sentences mentioning the searched term, or all |
| `pubmed.min_length` | `int` | `3` | Shorter candidates are dropped |
| `pubmed.min_similarity`, `pubmed.max_candidates` | `float`, `int` | `0.85`, `10` | Per concept: the candidates at least this similar to the label, the most similar `max_candidates` |
| `conceptnet.dump` | `str` | `"./data/input/conceptnet/conceptnet-assertions-5.7.0.csv.gz"` | The ConceptNet assertions dump |
| `conceptnet.database` | `str` | `"./data/input/conceptnet/conceptnet.sqlite"` | The database `SynGenConceptNetDB.py` builds; it records its relations and languages, and the stage refuses one built with others |
| `conceptnet.relations` | list | `["Synonym", "FormOf"]` | The relations followed, in both directions |
| `conceptnet.languages` | list | `["en"]` | The languages of the database; the first is that of the HPO |
| `conceptnet.hops` | `int` | `1` | Steps along the relations |
| `conceptnet.max_span` | `int` | `3` | The longest run of words looked up as one concept |
| `conceptnet.blacklisted_senses` | list | `[]` | ConceptNet senses (e.g. `food`) never followed |
| `conceptnet.seed_sources` | list | `["label", "llm", "pubmed"]` | The terms whose words are replaced |
| `conceptnet.max_substitutions` | `int` | `2` | Words replaced at once |
| `conceptnet.max_terms_per_seed` | `int` | `200` | The most new terms built from one term |
| `conceptnet.filter` | `bool` | `True` | Whether the LLM keeps only the neighbours that can replace a word in the concept |
| `conceptnet.filter_temperature` | `float` | `0.01` | Temperature of the filter |
| `conceptnet.filter_chunk` | `int` | `20` | Neighbours per filter prompt |
| `conceptnet.min_similarity`, `conceptnet.max_candidates` | `float`, `int` | `0.85`, `10` | As for PubMed |
| `embedding_model_names` | mapping | `{}` | Name of the embedding model in the column names; by default its ID without the organisation |
| `evaluation.embedding_match` | `float` | `1.0` | The cosine similarity of the embedding match; the accuracy of the embedding match needs 1 |
| `evaluation.threshold_lower`, `_upper`, `_step` | `float` | `0.5`, `1.0`, `0.01` | The thresholds of the similarity curve |
| `evaluation.bootstrap_samples` | `int` | `1000` | Bootstrap samples (over concepts) of the 95% intervals |
| `evaluation.bootstrap_seed` | `int` | `0` | Seed of the bootstrap samples |

### Settings of an experiment

| Key | Default | Description |
|---|---|---|
| `test` | off | Generate for the development concepts (`hpo.development.csv`); without it for all other concepts (`hpo.evaluation.csv`). The few-shot concepts never |
| `few_shot` | off | Add the few-shot examples (`hpo.fewshot.json`) to the generation prompt |
| `definition`, `comment`, `structure` | off | Add the definition, the comment, the labels of the parents and children to the generation prompt |
| `stages` | `["llm", "pubmed", "conceptnet"]` | The stages run; they start with `llm` and keep this order, e.g. `["llm", "conceptnet"]` |

The settings name the files of an experiment, e.g. `pubmed_Test_Def_Com_Str_FS_R5_LP_<model>.csv` (`Def`/`Com`/`Str`: definition, comment and structure in the prompt; `FS`: few-shot examples; `R5`: five rounds; `LP`: the stages up to this one; `No...` if off).
