#!/bin/bash
#SBATCH --job-name=syngen-filter
#SBATCH --partition=batch_long
#SBATCH --gres=gpu:a40:2                        # Fordert 2x NVIDIA A40 an
#SBATCH --time=7-00:00:00                       # Maximale Laufzeit (Format: D-HH:MM:SS)
#SBATCH --output=./data/slurmlogs/job_%j.out        # Speicherort für Ausgaben (%j = Job-ID)
#SBATCH --error=./data/slurmlogs/job_%j.err         # Speicherort für Fehlermeldungen

# Step 3 of 5: the LLM keeps the ConceptNet neighbours that can replace a
# span, for every experiment of syngen.experiments and every model of
# llm.model_id; nothing if syngen.conceptnet.filter is off.
#
# Environment: model package. Run: sbatch 3_filter_llm.sh
# See 1_generate_llm.sh for all steps.

# Sample with PyTorch: FlashInfer compiles its kernels with the system's nvcc,
# which is too old ("nvcc fatal : Unknown option '--compress-mode=size'").
export VLLM_USE_FLASHINFER_SAMPLER=0

EXPERIMENTS=$(python3 ./src/config.py) || exit 1
FAILED=0
for ((INDEX = 0; INDEX < EXPERIMENTS; INDEX++)); do
    python3 ./src/SynGenConceptNet.py "$INDEX" --filter || FAILED=1
done
exit $FAILED
