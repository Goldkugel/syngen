#!/bin/bash

clear

# start.sh [<step>]: one step of the generation (llm if none is given) for
# every experiment of syngen.experiments. The steps run in this order, each
# once the step before has finished for all experiments:
#   llm      generate_llm.sh              LLM generation (Slurm jobs)
#   sources  generate_sources.sh          marking, PubMed, ConceptNet spans
#   filter   filter_llm.sh                LLM filter (Slurm jobs)
#   build    generate_sources.sh --build  ConceptNet terms
# The LLM steps submit one Slurm job per experiment; sources and build run
# here, one experiment after the other. Without syngen.conceptnet.filter,
# sources also builds the ConceptNet terms and filter and build are not
# needed. Start each step from the Python environment of its script (model
# package or normalizer); sbatch passes it on to the jobs.
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
FAILED=0
for ((INDEX = 0; INDEX < EXPERIMENTS; INDEX++)); do
    if [ "$SCRIPT" = generate_sources.sh ]; then
        # A failing experiment does not stop the others.
        ./generate_sources.sh "$INDEX" "${OPTION[@]}" || FAILED=$((FAILED + 1))
    else
        sbatch "$SCRIPT" "$INDEX"
    fi
done

if [ $FAILED -gt 0 ]; then
    echo "start.sh: $FAILED of $EXPERIMENTS experiment(s) failed; see data/logs/." >&2
    exit 1
fi
