#!/bin/bash

# Step 5 of 5: normalises and embeds the synonyms and evaluates them, for
# every experiment of syngen.experiments and every model of llm.model_id.
#
# Environment: normalizer. Run: ./5_evaluate.sh
# Not a Slurm job. See 1_generate_llm.sh for all steps.

EXPERIMENTS=$(python3 ./src/config.py) || exit 1
FAILED=0
for ((INDEX = 0; INDEX < EXPERIMENTS; INDEX++)); do
    python3 ./src/SynGenEmbed.py "$INDEX" &&
        python3 ./src/SynGenEval.py "$INDEX" || FAILED=1
done
exit $FAILED
