#!/bin/bash
#SBATCH --job-name=syngen-llm
#SBATCH --partition=batch_long
#SBATCH --gres=gpu:a40:2                        # Fordert 2x NVIDIA A40 an
#SBATCH --time=7-00:00:00                       # Maximale Laufzeit (Format: D-HH:MM:SS)
#SBATCH --output=./data/slurmlogs/job_%j.out        # Speicherort für Ausgaben (%j = Job-ID)
#SBATCH --error=./data/slurmlogs/job_%j.err         # Speicherort für Fehlermeldungen

# Step 1 of 5: the LLM generates the synonyms (LLM stage) for every experiment
# of syngen.experiments and every model of llm.model_id.
#
# Environment: model package. Run: sbatch 1_generate_llm.sh
#
# The steps, each started once the one before has finished:
#   1  sbatch 1_generate_llm.sh   LLM generation           model
#   2  ./2_generate_sources.sh    PubMed, ConceptNet spans  normalizer
#   3  sbatch 3_filter_llm.sh     LLM filter of ConceptNet  model
#   4  ./4_build_conceptnet.sh    ConceptNet terms          normalizer
#   5  ./5_evaluate.sh            embedding and evaluation  normalizer

# Sample with PyTorch: FlashInfer compiles its kernels with the system's nvcc,
# which is too old ("nvcc fatal : Unknown option '--compress-mode=size'").
export VLLM_USE_FLASHINFER_SAMPLER=0

EXPERIMENTS=$(python3 ./src/config.py) || exit 1
FAILED=0
for ((INDEX = 0; INDEX < EXPERIMENTS; INDEX++)); do
    python3 ./src/SynGenLLM.py "$INDEX" || FAILED=1
done
exit $FAILED
