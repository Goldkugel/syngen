#!/bin/bash

# generate_sources.sh <index> [<model ID>] [--build]: the PubMed and
# ConceptNet stages of the experiment, for the synonyms of every model of
# llm.model_id (or only the given one). Not a Slurm job: it needs no GPU (the
# NER uses one if there is one) but internet access for PubMed, so run it
# directly, e.g. on the login node, from the Python environment of the
# normalizer (see generate_llm.sh for the order of the scripts).
#   without --build  marks the synonyms of the LLM stage that equal their
#                    label, runs the PubMed stage and finds the spans of the
#                    ConceptNet stage; without syngen.conceptnet.filter it
#                    also builds the ConceptNet terms, which ends the stage
#   --build          after filter_llm.sh: builds and ranks the ConceptNet
#                    terms

BUILD=0
ARGUMENTS=()
for ARGUMENT in "$@"; do
    if [ "$ARGUMENT" = "--build" ]; then
        BUILD=1
    else
        ARGUMENTS+=("$ARGUMENT")
    fi
done
if [ ${#ARGUMENTS[@]} -lt 1 ]; then
    echo "Usage: generate_sources.sh <index> [<model ID>] [--build]" >&2
    exit 1
fi

if [ $BUILD -eq 1 ]; then
    python3 ./src/SynGenConceptNet.py "${ARGUMENTS[@]}" --build
else
    python3 ./src/SynGenLLM.py "${ARGUMENTS[@]}" --mark || exit 1
    python3 ./src/SynGenPubMed.py "${ARGUMENTS[@]}" || exit 1
    python3 ./src/SynGenConceptNet.py "${ARGUMENTS[@]}" --spans
fi
