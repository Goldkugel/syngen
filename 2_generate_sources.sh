#!/bin/bash

# Step 2 of 5: the PubMed stage and the spans of the ConceptNet stage, for
# every experiment of syngen.experiments and every model of llm.model_id.
#
# Environment: normalizer. Run: ./2_generate_sources.sh
# Not a Slurm job: it needs no GPU but internet access for PubMed, e.g. on
# the login node. See 1_generate_llm.sh for all steps.

EXPERIMENTS=$(python3 ./src/config.py) || exit 1
FAILED=0
for ((INDEX = 0; INDEX < EXPERIMENTS; INDEX++)); do
    python3 ./src/SynGenPubMed.py "$INDEX" &&
        python3 ./src/SynGenConceptNet.py "$INDEX" --spans || FAILED=1
done
exit $FAILED
