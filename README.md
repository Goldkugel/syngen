# syngen

Generates synonyms of [Human Phenotype Ontology](https://hpo.jax.org/) (HPO) concepts and evaluates them against the exact synonyms HPO lists. Three sources are used one after the other, each starting from the synonyms of the one before: a local LLM, PubMed and ConceptNet.

## Features

- **Three stages** — the **LLM** generates synonyms from the label, definition, comment and the labels of the parents and children of a concept; **PubMed** is searched with the label and every LLM synonym, and biomedical terms in the sentences mentioning them become candidates; **ConceptNet** replaces the words of the label and of the synonyms found so far by their neighbours (e.g. `Synonym`, `FormOf`). PubMed and ConceptNet keep per concept the candidates most similar to the label.
- **Repeated generation** — every concept is prompted `syngen.llm.rounds` times with its own seed per round; a synonym records in how many rounds it was generated.
- **Experiments in the configuration** — every experiment of `syngen.experiments` in [`config/config.yaml`](config/config.yaml) sets which information about a concept the prompt contains, whether it contains few-shot examples, which stages run, and whether the development set or the evaluation set (all other concepts) is used.
- **Gold standard of exact synonyms only** — related, broad and narrow synonyms, synonyms also listed with another class, synonyms equal to their label and obsolete synonyms are left out.
- **Three matches** — a generated synonym matches a gold synonym `lexical`ly (the same text, case ignored), `normalized` (the same tokens of the [`normalizer`](https://github.com/Goldkugel/normalizer), so word order, inflection and stop words do not matter) or by `embedding` (cosine similarity 1, SapBERT via the [`embedder`](https://github.com/Goldkugel/embedder)); a synonym equal to its label is reported as `as_label`.
- **Evaluation** — per stage on its own and cumulatively: precision, recall and F1 (micro with bootstrap intervals, and macro), accuracy (the gold synonyms generated for any concept), the words covered (Jaccard index of the words of gold and generated synonyms), a similarity threshold curve and a curve over the number of rounds.
- **Resumable and reproducible** — the [`model`](https://github.com/Goldkugel/model) package stores every prompt and answer; PubMed searches (up to a fixed publication date), articles and NER spans are cached in SQLite; outputs are made again only when their input is newer or their settings changed (`<output>.settings.json`).

## Repository Structure

```
syngen/
├── config/
│   ├── config.yaml          # Configuration: syngen, adapter, logger, llm, embedder and normalizer sections
│   └── README.md
├── data/
│   ├── input/               # hp.owl, the ConceptNet dump and database
│   ├── cache/               # PubMed cache and embedding cache (not tracked)
│   ├── logs/                # Log files, prompt logs and run metadata
│   ├── output/
│   │   ├── transformed/     # The HPO, development and evaluation set, few-shot examples
│   │   ├── llm/             # Answers and synonyms of the LLM
│   │   ├── pubmed/          # Synonyms of the LLM and PubMed
│   │   ├── conceptnet/      # Synonyms of all stages
│   │   ├── embedded/        # Normalised and embedded generated and gold synonyms
│   │   └── evaluated/       # Evaluation summary, curves and plots
│   ├── slurmlogs/           # Output of the Slurm jobs
│   └── tmp/                 # Answer stores of the model package (not tracked)
├── prompts/                 # Prompt templates of the generation and the ConceptNet filter
├── src/
│   ├── SynGenLLM.py         # Preparation run and LLM stage
│   ├── SynGenPubMed.py      # PubMed stage
│   ├── SynGenConceptNet.py  # ConceptNet stage
│   ├── SynGenConceptNetDB.py # Builds the ConceptNet database
│   ├── SynGenEmbed.py       # Normalises and embeds the synonyms
│   ├── SynGenEval.py        # Evaluates the synonyms
│   ├── SynGenDevelopmentSet.py # Draws the development and evaluation set
│   ├── SynGenUtils.py       # Helpers shared by the scripts
│   ├── SynGenConfig.py      # Schema of the syngen section
│   ├── config.py            # Configuration, experiment, column and file names
│   └── README.md
├── tests/                   # Unit tests (pytest)
├── start.sh                 # Preparation run, then one job per experiment for a step
├── generate_llm.sh          # Slurm job: LLM generation of one experiment
├── generate_sources.sh      # PubMed and ConceptNet stages of one experiment (no Slurm job)
├── filter_llm.sh            # Slurm job: LLM filter of the ConceptNet stage of one experiment
├── finalize.sh              # Embeds and evaluates every experiment
├── cleanUp.sh               # Deletes logs, synonyms and evaluations
├── pyproject.toml
└── LICENSE
```

## Requirements & Installation

Python 3.10 or newer, and GPUs for the LLM (the `model` package runs it with [vLLM](https://github.com/vllm-project/vllm); `Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4` fits on two A40). The dependencies, including the sibling packages [`logger`](https://github.com/Goldkugel/logger), [`adapter`](https://github.com/Goldkugel/adapter), [`embedder`](https://github.com/Goldkugel/embedder), [`model`](https://github.com/Goldkugel/model) and [`normalizer`](https://github.com/Goldkugel/normalizer) at pinned versions, are declared in [`pyproject.toml`](pyproject.toml).

The `model` package (vLLM) and the `normalizer` (spaCy 3.7) need library versions that cannot be installed in the same environment, so syngen needs two Python environments, e.g. two virtual environments:

```bash
python3 -m venv ~/venvs/syngen-llm
~/venvs/syngen-llm/bin/pip install -e ".[llm]"
python3 -m venv ~/venvs/syngen-normalizer
~/venvs/syngen-normalizer/bin/pip install -e ".[normalizer]"
```

The scripts that run the LLM (`generate_llm.sh`, `filter_llm.sh`) use the first, all others (`generate_sources.sh`, `finalize.sh`, the tests) the second. The scripts do not activate an environment: activate it before starting a script; `sbatch` passes it on to the jobs of the LLM scripts.

Inputs (see [`data/input/README.md`](data/input/README.md)):

1. The HPO release at `data/input/hpo/hp.owl`.
2. The ConceptNet 5.7 assertions at `data/input/conceptnet/conceptnet-assertions-5.7.0.csv.gz`; then build the database once with `python3 ./src/SynGenConceptNetDB.py`.
3. Optionally an NCBI API key in the environment variable `NCBI_API_KEY` (10 instead of 3 PubMed requests per second); never put it into the configuration.

## Usage

All scripts are run from the repository root.

1. **Generate:** the generation runs in four steps; `./start.sh <step>` runs the preparation (transforms the HPO, draws the development and evaluation set and the few-shot examples, each only if needed) and runs the step for every experiment, with every model: the LLM steps as one Slurm job per experiment, `sources` and `build` directly (they need no GPU but internet access for PubMed, e.g. on the login node), one experiment after the other. Start a step once the step before has finished for all experiments, from the environment of its script:

   | Step | Environment | Script | What it does |
   |---|---|---|---|
   | `./start.sh llm` | LLM | `generate_llm.sh` | LLM generation |
   | `./start.sh sources` | normalizer | `generate_sources.sh` | Marks the LLM synonyms equal to their label, the PubMed stage, the spans of the ConceptNet stage; without `syngen.conceptnet.filter` also the ConceptNet terms, which ends the generation |
   | `./start.sh filter` | LLM | `filter_llm.sh` | The LLM filters the ConceptNet neighbours |
   | `./start.sh build` | normalizer | `generate_sources.sh --build` | The ConceptNet stage: builds and ranks the new terms |

   A single experiment runs with `sbatch generate_llm.sh <index> [<model ID>]`, `./generate_sources.sh <index> [<model ID>] [--build]` and `sbatch filter_llm.sh <index> [<model ID>]`.
2. **Evaluate:** when the jobs are done, `./finalize.sh` (normalizer environment) embeds and evaluates the synonyms of every experiment.

| Command | What it does |
|---|---|
| `python3 ./src/SynGenLLM.py` | Preparation run: transforms the HPO, draws the development and evaluation set and the few-shot examples, each only if needed |
| `python3 ./src/SynGenLLM.py <index> [<model ID>]` | LLM stage of experiment `syngen.experiments[<index>]` (from 0) with every model of `llm.model_id` (or the given one): generates the synonyms (LLM environment) |
| `python3 ./src/SynGenLLM.py <index> [<model ID>] --mark` | Marks the synonyms equal to their label and writes the file of the LLM stage (normalizer environment) |
| `python3 ./src/SynGenPubMed.py <index> [<model ID>] [--fetch]` | PubMed stage; with `--fetch` only searches and fetches the articles |
| `python3 ./src/SynGenConceptNet.py <index> [<model ID>] --spans\|--filter\|--build` | ConceptNet stage in three steps: the spans and their neighbours (normalizer), the LLM filter (LLM), the new terms (normalizer); without `syngen.conceptnet.filter`, `--spans` builds the terms as well |
| `python3 ./src/SynGenConceptNetDB.py` | Builds the ConceptNet database from the dump |
| `python3 ./src/SynGenEmbed.py <index> [<model ID>]` | Normalises and embeds the generated and gold synonyms |
| `python3 ./src/SynGenEval.py <index> [<model ID>]` | Evaluates the synonyms |
| `python3 ./src/SynGenDevelopmentSet.py` | Draws the development and evaluation set again (e.g. after changing the seed) |
| `python3 ./src/config.py` | Prints the number of experiments |

`./cleanUp.sh` deletes the logs, the synonyms, the evaluations and the answer stores; the transformed HPO, the caches and the ConceptNet database are kept.

## Configuration

[`config/config.yaml`](config/config.yaml) has one section per part: `syngen` (the experiments and syngen's own settings, see [`config/README.md`](config/README.md)), `adapter` (reading `hp.owl`), `logger`, `llm` (the model and the prompts), `embedder` (the embedding model of the similarities) and `normalizer`. Paths are relative to the repository root.

After the first test run, set `syngen.pubmed.min_similarity` and `syngen.conceptnet.min_similarity` on the development set, e.g. from the similarity threshold curve of `SynGenEval.py`.

## Pipeline

| Step | Script | Output |
|---|---|---|
| Transformation | `SynGenLLM.py` | [`data/output/transformed/`](data/output/transformed/README.md): `hp.owl` in the adapter's format |
| Development and evaluation set | `SynGenLLM.py`, `SynGenDevelopmentSet.py` | [`data/output/transformed/`](data/output/transformed/README.md): concepts drawn at random until they have 500 gold synonyms, the rest, and the few-shot examples |
| LLM | `SynGenLLM.py <index>`, then `--mark` | [`data/output/llm/`](data/output/llm/README.md) |
| PubMed | `SynGenPubMed.py <index>` | [`data/output/pubmed/`](data/output/pubmed/README.md) |
| ConceptNet | `SynGenConceptNet.py <index> --spans`, `--filter`, `--build` | [`data/output/conceptnet/`](data/output/conceptnet/README.md) |
| Embedding | `SynGenEmbed.py <index>` | [`data/output/embedded/`](data/output/embedded/README.md) |
| Evaluation | `SynGenEval.py <index>` | [`data/output/evaluated/`](data/output/evaluated/README.md) |

The prompts are described in [`prompts/README.md`](prompts/README.md), the logs in [`data/logs/README.md`](data/logs/README.md), the tests in [`tests/README.md`](tests/README.md).

The PubMed and ConceptNet stages build on two bachelor's theses, `ba-synonym-generation` (PubMed, NER candidates) and `ba-syngen-conceptnet-llm` (ConceptNet expansion, LLM filter, recombination); their code was reimplemented, not imported.

Version 1 of syngen (the earlier scripts, including those now in [synclass](https://github.com/Goldkugel/synclass) and [syntype](https://github.com/Goldkugel/syntype)) is tagged `v1.0.0`.

## License

GNU General Public License v3.0, see [`LICENSE`](LICENSE).
