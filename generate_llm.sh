#!/bin/bash
#SBATCH --job-name=syngen-llm
#SBATCH --partition=batch_long
#SBATCH --gres=gpu:a40:2                        # Fordert 2x NVIDIA A40 an
#SBATCH --time=7-00:00:00                       # Maximale Laufzeit (Format: D-HH:MM:SS)
#SBATCH --output=./data/slurmlogs/job_%j.out        # Speicherort für Ausgaben (%j = Job-ID)
#SBATCH --error=./data/slurmlogs/job_%j.err         # Speicherort für Fehlermeldungen

# generate_llm.sh <step> <index> [<model ID>]: the steps of the experiment
# that run the LLM (model package), with every model of llm.model_id (or only
# the given one). Submit it from the Python environment of the model package;
# sbatch passes it on to the job. The steps of an experiment run in the order
# llm, pubmed, filter, conceptnet (start.sh <step>); pubmed and conceptnet need
# the normalizer (generate_normalizer.sh).
#   llm     the LLM stage: generates the synonyms, unmarked
#   filter  the ConceptNet stage: the LLM filters the neighbours of the spans
#           (nothing if syngen.conceptnet.filter is off)

# Sample with PyTorch: FlashInfer compiles its kernels with the system's nvcc,
# which is too old ("nvcc fatal : Unknown option '--compress-mode=size'").
export VLLM_USE_FLASHINFER_SAMPLER=0

if [ $# -lt 2 ]; then
    echo "Usage: generate_llm.sh <llm|filter> <index> [<model ID>]" >&2
    exit 1
fi
STEP=$1
shift

case "$STEP" in
    llm)    python3 ./src/SynGenLLM.py "$@" ;;
    filter) python3 ./src/SynGenConceptNet.py "$@" --filter ;;
    *)      echo "generate_llm.sh: unknown step '$STEP' (llm or filter)." >&2; exit 1 ;;
esac
