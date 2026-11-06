# Tests

Unit tests of the helpers, the PubMed client and cache, the ConceptNet database and recombination, and the evaluation, without GPU, network or model downloads (the normalizer's `en_core_sci_sm` model is needed). Run from the repository root:

```bash
pip install -e ".[test]"
pytest
```

`conftest.py` runs them in a temporary folder with a copy of `config/` and `prompts/`, so that logs and outputs do not land in `data/`.
