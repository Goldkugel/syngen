#!/bin/bash

# build_conceptnet.sh [<index> [<model ID>]]: the last step of the ConceptNet
# stage of the experiment (syngen.experiments[<index>], 0 if none is given),
# for the synonyms of every model of llm.model_id (or only the given one):
# replaces the spans by the neighbours the LLM kept (filter_llm.sh), ranks
# the new terms and writes the file of the stage. Not a Slurm job: run it
# directly from the Python environment of the normalizer (see generate_llm.sh
# for the order of the scripts).

if [ $# -eq 0 ]; then
    set -- 0
fi

python3 ./src/SynGenConceptNet.py "$@" --build
