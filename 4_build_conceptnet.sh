#!/bin/bash

# Step 4 of 5: the ConceptNet stage builds and ranks the new terms, for every
# experiment of syngen.experiments and every model of llm.model_id.
#
# Environment: normalizer. Run: ./4_build_conceptnet.sh
# Not a Slurm job. See 1_generate_llm.sh for all steps.

EXPERIMENTS=$(python3 ./src/config.py) || exit 1
FAILED=0
for ((INDEX = 0; INDEX < EXPERIMENTS; INDEX++)); do
    python3 ./src/SynGenConceptNet.py "$INDEX" --build || FAILED=1
done
exit $FAILED
