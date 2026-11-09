#!/bin/bash

clear

# start.sh [<step>]: one Slurm job per experiment of syngen.experiments for
# one step of the generation (llm if none is given). The steps run in this
# order, each once all jobs of the step before have finished:
#   llm         generate_llm.sh         the LLM stage
#   pubmed      generate_normalizer.sh  marking, PubMed stage, ConceptNet spans
#   filter      generate_llm.sh         the LLM filter of the ConceptNet stage
#   conceptnet  generate_normalizer.sh  the ConceptNet stage
# Start each step from the Python environment of its script (model package or
# normalizer); sbatch passes it on to the jobs.
STEP=${1:-llm}
case "$STEP" in
    llm|filter)        SCRIPT=generate_llm.sh ;;
    pubmed|conceptnet) SCRIPT=generate_normalizer.sh ;;
    *) echo "start.sh: unknown step '$STEP' (llm, pubmed, filter or conceptnet)." >&2; exit 1 ;;
esac

# Preparation: transforms the HPO, draws the development and evaluation set
# and the few-shot examples, each only if outdated.
python3 ./src/SynGenLLM.py || exit 1

EXPERIMENTS=$(python3 ./src/config.py) || exit 1
for ((INDEX = 0; INDEX < EXPERIMENTS; INDEX++)); do
    sbatch "$SCRIPT" "$STEP" "$INDEX"
done
