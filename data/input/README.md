# Input

The inputs of syngen; the files are not tracked.

| File | Read by | Content |
|---|---|---|
| `hpo/hp.owl` | the preparation run (`python3 ./src/SynGenLLM.py`) with the `adapter` package (`adapter.hpo` in [`config/config.yaml`](../../config/config.yaml)) | The Human Phenotype Ontology in OWL, e.g. from the [HPO releases](https://github.com/obophenotype/human-phenotype-ontology/releases) |
| `conceptnet/conceptnet-assertions-5.7.0.csv.gz` | `SynGenConceptNetDB.py` (`syngen.conceptnet.dump`) | The ConceptNet 5.7 assertions, from the [ConceptNet downloads](https://github.com/commonsense/conceptnet5/wiki/Downloads); packed or unpacked |
| `conceptnet/conceptnet.sqlite` | `SynGenConceptNet.py` (`syngen.conceptnet.database`) | The ConceptNet database built by `python3 ./src/SynGenConceptNetDB.py` from the dump, with only the relations and languages of `syngen.conceptnet` |

When `hp.owl` is replaced by a newer one, the next preparation run transforms the HPO and draws the development set again.
