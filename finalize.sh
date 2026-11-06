#!/bin/bash

clear

# Brings the transformed HPO and the sets up to date.
python3 ./src/SynGenLLM.py || exit 1

# Embeds and evaluates the synonyms of every experiment.
EXPERIMENTS=$(python3 ./src/config.py) || exit 1
for ((INDEX = 0; INDEX < EXPERIMENTS; INDEX++)); do
    # An experiment without synonyms is not evaluated.
    python3 ./src/SynGenEmbed.py "$INDEX" && python3 ./src/SynGenEval.py "$INDEX"
done
