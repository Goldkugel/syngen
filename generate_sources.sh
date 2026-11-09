#!/bin/bash
#SBATCH --job-name=syngen-sources
#SBATCH --partition=batch_long
#SBATCH --gres=gpu:a40:1                        # NER of PubMed and SapBERT: 1x NVIDIA A40
#SBATCH --time=7-00:00:00                       # Maximale Laufzeit (Format: D-HH:MM:SS)
#SBATCH --output=./data/slurmlogs/job_%j.out        # Speicherort für Ausgaben (%j = Job-ID)
#SBATCH --error=./data/slurmlogs/job_%j.err         # Speicherort für Fehlermeldungen

# generate_sources.sh <index> [<model ID>] [--build]: the PubMed and
# ConceptNet stages of the experiment, for the synonyms of every model of
# llm.model_id (or only the given one). Submit it from the Python environment
# of the normalizer; sbatch passes it on to the job (see generate_llm.sh for
# the order of the scripts).
#   without --build  marks the synonyms of the LLM stage that equal their
#                    label, runs the PubMed stage and finds the spans of the
#                    ConceptNet stage; without syngen.conceptnet.filter it
#                    also builds the ConceptNet terms, which ends the stage
#   --build          after filter_llm.sh: builds and ranks the ConceptNet
#                    terms
#
# The PubMed stage needs the search results and abstracts in its cache: if
# the compute nodes have no internet access, run
#   python3 ./src/SynGenLLM.py <index> --mark
#   python3 ./src/SynGenPubMed.py <index> --fetch
# on the login node after generate_llm.sh first.

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
