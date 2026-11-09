#!/bin/bash

clear

# start.sh [<step>]: one Slurm job per experiment of syngen.experiments for
# one step of the generation (llm if none is given). The steps run in this
# order, each once all jobs of the step before have finished:
#   llm      generate_llm.sh              LLM generation
#   sources  generate_sources.sh          marking, PubMed, ConceptNet spans
#   filter   filter_llm.sh                LLM filter of the ConceptNet stage
#   build    generate_sources.sh --build  ConceptNet terms
# Without syngen.conceptnet.filter, sources also builds the ConceptNet terms
# and filter and build are not needed. Start each step from the Python
# environment of its script (model package or normalizer); sbatch passes it
# on to the jobs.
STEP=${1:-llm}
OPTION=()
case "$STEP" in
    llm)     SCRIPT=generate_llm.sh ;;
    sources) SCRIPT=generate_sources.sh ;;
    filter)  SCRIPT=filter_llm.sh ;;
    build)   SCRIPT=generate_sources.sh; OPTION=(--build) ;;
    *) echo "start.sh: unknown step '$STEP' (llm, sources, filter or build)." >&2; exit 1 ;;
esac

# Preparation: transforms the HPO, draws the development and evaluation set
# and the few-shot examples, each only if outdated.
python3 ./src/SynGenLLM.py || exit 1

EXPERIMENTS=$(python3 ./src/config.py) || exit 1
for ((INDEX = 0; INDEX < EXPERIMENTS; INDEX++)); do
    sbatch "$SCRIPT" "$INDEX" "${OPTION[@]}"
done
