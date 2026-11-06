# Evaluated

Written by `SynGenEval.py <index>`, per experiment and model (`eval_<experiment>_<stages>_<model>_...`).

| File | Content |
|---|---|
| `..._summary.csv` | One row per view (each stage alone, and cumulative: with the stages before it) and match (`lexical`, `normalized`, `embedding`): the counts, micro precision, recall and F1 with their 95% bootstrap intervals, macro precision, recall and F1, accuracy (gold synonyms generated for any concept), the words covered (Jaccard and share of the gold words, also without the label's), the synonyms equal to the label (`as_label`), those HPO lists with another class (`hpo_related`, ...) and those naming another concept (`other_concept`) |
| `..._summary.png` | Precision, recall and F1 of every view and match |
| `..._threshold_curve.csv`, `.png` | Precision, recall and F1 of every view for the embedding match at the thresholds of `syngen.evaluation` |
| `..._rounds_curve.csv`, `.png` | Precision, recall and F1 (normalised match) of the LLM synonyms generated in at least k rounds |

The final evaluation is the cumulative one of the last stage.
