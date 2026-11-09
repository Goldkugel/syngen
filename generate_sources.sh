#!/bin/bash

# generate_sources.sh [<index> [<model ID>]]: the PubMed stage and the first
# step of the ConceptNet stage of the experiment (syngen.experiments[<index>],
# 0 if none is given), for the synonyms of every model of llm.model_id (or
# only the given one): marks the synonyms of the LLM stage that equal their
# label, runs the PubMed stage and finds the spans of the ConceptNet stage;
# without syngen.conceptnet.filter it also builds the ConceptNet terms, which
# ends the generation. Not a Slurm job: it needs no GPU (the NER uses one if
# there is one) but internet access for PubMed, so run it directly, e.g. on
# the login node, from the Python environment of the normalizer (see
# generate_llm.sh for the order of the scripts).

if [ $# -eq 0 ]; then
    set -- 0
fi

python3 ./src/SynGenLLM.py "$@" --mark || exit 1
python3 ./src/SynGenPubMed.py "$@" || exit 1
python3 ./src/SynGenConceptNet.py "$@" --spans
