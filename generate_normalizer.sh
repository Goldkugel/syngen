#!/bin/bash
#SBATCH --job-name=syngen-normalizer
#SBATCH --partition=batch_long
#SBATCH --gres=gpu:a40:1                        # NER of PubMed and SapBERT: 1x NVIDIA A40
#SBATCH --time=7-00:00:00                       # Maximale Laufzeit (Format: D-HH:MM:SS)
#SBATCH --output=./data/slurmlogs/job_%j.out        # Speicherort für Ausgaben (%j = Job-ID)
#SBATCH --error=./data/slurmlogs/job_%j.err         # Speicherort für Fehlermeldungen

# generate_normalizer.sh <step> <index> [<model ID>]: the steps of the
# experiment that need the normalizer, for the synonyms of every model of
# llm.model_id (or only the given one). Submit it from the Python environment
# of the normalizer; sbatch passes it on to the job. The steps of an
# experiment run in the order llm, pubmed, filter, conceptnet (start.sh
# <step>); llm and filter run the LLM (generate_llm.sh).
#   pubmed      marks the synonyms of the LLM stage that equal their label,
#               runs the PubMed stage and finds the spans of the ConceptNet
#               stage
#   conceptnet  the ConceptNet stage: builds and ranks the new terms
#
# The PubMed stage needs the search results and abstracts in its cache: if
# the compute nodes have no internet access, run
#   python3 ./src/SynGenLLM.py <index> --mark
#   python3 ./src/SynGenPubMed.py <index> --fetch
# on the login node after the llm step first.

if [ $# -lt 2 ]; then
    echo "Usage: generate_normalizer.sh <pubmed|conceptnet> <index> [<model ID>]" >&2
    exit 1
fi
STEP=$1
shift

case "$STEP" in
    pubmed)
        python3 ./src/SynGenLLM.py "$@" --mark || exit 1
        python3 ./src/SynGenPubMed.py "$@" || exit 1
        python3 ./src/SynGenConceptNet.py "$@" --spans ;;
    conceptnet)
        python3 ./src/SynGenConceptNet.py "$@" --build ;;
    *)
        echo "generate_normalizer.sh: unknown step '$STEP' (pubmed or conceptnet)." >&2; exit 1 ;;
esac
