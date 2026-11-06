#!/bin/bash
#SBATCH --job-name=syngen
#SBATCH --partition=batch_long
#SBATCH --gres=gpu:a40:2                        # Fordert 2x NVIDIA A40 an
#SBATCH --time=7-00:00:00                       # Maximale Laufzeit (Format: D-HH:MM:SS)
#SBATCH --output=./data/slurmlogs/job_%j.out        # Speicherort für Ausgaben (%j = Job-ID)
#SBATCH --error=./data/slurmlogs/job_%j.err         # Speicherort für Fehlermeldungen

# generate.sh <index> [<model ID>]: the stages of the experiment with every
# model of llm.model_id (or only the given one). The PubMed stage needs the
# search results and abstracts in its cache: if the compute nodes have no
# internet access, run "python3 ./src/SynGenPubMed.py <index> --fetch" on the
# login node after the LLM stage first, then this job again; the answers of
# the LLM stage are reused.

# Sample with PyTorch: FlashInfer compiles its kernels with the system's nvcc,
# which is too old ("nvcc fatal : Unknown option '--compress-mode=size'").
export VLLM_USE_FLASHINFER_SAMPLER=0

python3 ./src/SynGenLLM.py "$@" || exit 1
python3 ./src/SynGenPubMed.py "$@" || exit 1
python3 ./src/SynGenConceptNet.py "$@" || exit 1
