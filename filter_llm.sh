#!/bin/bash
#SBATCH --job-name=syngen-filter
#SBATCH --partition=batch_long
#SBATCH --gres=gpu:a40:2                        # Fordert 2x NVIDIA A40 an
#SBATCH --time=7-00:00:00                       # Maximale Laufzeit (Format: D-HH:MM:SS)
#SBATCH --output=./data/slurmlogs/job_%j.out        # Speicherort für Ausgaben (%j = Job-ID)
#SBATCH --error=./data/slurmlogs/job_%j.err         # Speicherort für Fehlermeldungen

# filter_llm.sh [<index> [<model ID>]]: the LLM keeps the ConceptNet
# neighbours that can replace a span, for the experiment
# (syngen.experiments[<index>], 0 if none is given) and the synonyms of every
# model of llm.model_id (or only the given one); nothing if
# syngen.conceptnet.filter is off. Submit it from the Python environment of
# the model package after generate_sources.sh, then run build_conceptnet.sh
# (see generate_llm.sh for the order of the scripts).

# Sample with PyTorch: FlashInfer compiles its kernels with the system's nvcc,
# which is too old ("nvcc fatal : Unknown option '--compress-mode=size'").
export VLLM_USE_FLASHINFER_SAMPLER=0

if [ $# -eq 0 ]; then
    set -- 0
fi

python3 ./src/SynGenConceptNet.py "$@" --filter
