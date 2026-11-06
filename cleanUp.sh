#!/bin/bash

# Removes the outputs of the runs; the transformed HPO, the caches (PubMed
# articles, embeddings) and the ConceptNet database take long to make and are
# kept.
rm -f ./data/logs/*.log
rm -f ./data/output/llm/*.csv ./data/output/pubmed/*.csv ./data/output/conceptnet/*.csv
rm -f ./data/output/embedded/*.csv ./data/output/embedded/*.json
rm -f ./data/output/evaluated/*.png ./data/output/evaluated/*.csv
rm -f ./data/tmp/*.json ./data/tmp/*.lock
rm -f ./data/slurmlogs/*.err
rm -f ./data/slurmlogs/*.out

exit 0
