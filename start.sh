#!/bin/bash

clear

# Preparation: transforms the HPO, draws the development and evaluation set
# and the few-shot examples, each only if outdated.
python3 ./src/SynGenLLM.py || exit 1

# One generation job per experiment of syngen.experiments: the LLM, PubMed
# and ConceptNet stages one after the other.
EXPERIMENTS=$(python3 ./src/config.py) || exit 1
for ((INDEX = 0; INDEX < EXPERIMENTS; INDEX++)); do
    sbatch generate.sh "$INDEX"
done
