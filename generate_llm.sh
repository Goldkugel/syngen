#!/bin/bash
#SBATCH --job-name=syngen-llm
#SBATCH --partition=batch_long
#SBATCH --gres=gpu:a40:2                        # Fordert 2x NVIDIA A40 an
#SBATCH --time=7-00:00:00                       # Maximale Laufzeit (Format: D-HH:MM:SS)
#SBATCH --output=./data/slurmlogs/job_%j.out        # Speicherort für Ausgaben (%j = Job-ID)
#SBATCH --error=./data/slurmlogs/job_%j.err         # Speicherort für Fehlermeldungen

# generate_llm.sh <index> [<model ID>]: the LLM stage of the experiment with
# every model of llm.model_id (or only the given one): generates the
# synonyms, unmarked. Submit it from the Python environment of the model
# package; sbatch passes it on to the job. The scripts of an experiment run
# in this order (start.sh <step>):
#   generate_llm.sh               LLM generation          (model)
#   generate_sources.sh           PubMed and ConceptNet   (normalizer)
#   filter_llm.sh                 LLM filter              (model)
#   generate_sources.sh --build   ConceptNet terms        (normalizer)
# Without syngen.conceptnet.filter the last two are not needed.

# Sample with PyTorch: FlashInfer compiles its kernels with the system's nvcc,
# which is too old ("nvcc fatal : Unknown option '--compress-mode=size'").
export VLLM_USE_FLASHINFER_SAMPLER=0

if [ $# -lt 1 ]; then
    echo "Usage: generate_llm.sh <index> [<model ID>]" >&2
    exit 1
fi

python3 ./src/SynGenLLM.py "$@"
