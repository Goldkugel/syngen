"""
Evaluates the generated synonyms of an experiment against the gold standard
(the exact synonyms of the HPO, see goldStandard() in SynGenUtils.py), for
every stage on its own and for the stages up to it together ("cumulative";
the cumulative evaluation of the last stage is the final one).

A generated synonym matches a gold synonym of its concept
- "lexical":    if both are the same text (lexicalKey(): case and white space
                ignored);
- "normalized": if both have the same normalised form (normalizer package:
                the same tokens after lemmatising and removing stop words);
- "embedding":  if their cosine similarity is at least
                syngen.evaluation.embedding_match (1: the same embedding).
A generated synonym equal to the label of its concept is counted as
"as_label" and not scored.

Per match: precision (generated synonyms that match a gold synonym of their
concept), recall (gold synonyms matched by a generated synonym of their
concept) and F1, micro (over all synonyms, with bootstrap intervals over the
concepts) and macro (the mean over the concepts); accuracy (the gold synonyms,
deduplicated over all concepts, generated for any concept); and the words
covered (per concept the Jaccard index of the words of the gold and of the
generated synonyms, and the share of gold words generated, also without the
words of the label). Only concepts with gold synonyms are scored.

    python3 ./src/SynGenEval.py <index> [<model>]
"""

import shlex
import sys

import numpy  as np
import pandas as pd

sys.dont_write_bytecode = True

from SynGenUtils import *
from config      import *

from logger      import Logger

l = Logger(config = configFile)

evaluationConfig = syngenConfig.evaluation

lexicalMatch    = "lexical"
normalizedMatch = "normalized"
embeddingMatch  = "embedding"
matchTypes      = [lexicalMatch, normalizedMatch, embeddingMatch]

aloneMode       = "alone"
cumulativeMode  = "cumulative"

# The similarity of two vectors equal up to rounding.
similarityTolerance = 1e-6

# ==============================================================================
# Matching
# ==============================================================================

def keysOf(texts : pd.Series, normalized : pd.Series, embeddingKeys : pd.Series, match : str) -> list:
    """
    The key a synonym is compared by in a match; "" never matches.
    """
    if match == lexicalMatch:
        return [lexicalKey(text) for text in texts]
    if match == normalizedMatch:
        return normalized.fillna("").astype(str).tolist()
    return embeddingKeys.fillna("").astype(str).tolist()

def viewStages(stage : str, mode : str) -> list:
    """The stages whose synonyms a view evaluates."""
    if mode == aloneMode:
        return [stage]
    return experimentStages[:experimentStages.index(stage) + 1]

def goldSimilarity(gold : pd.DataFrame, viewSources : list) -> np.ndarray:
    """
    The highest similarity of every gold synonym to the synonyms of the view;
    -inf where there is none.
    """
    columns = [stageSimilarityColumn(stage) for stage in viewSources]
    return gold[columns].astype(float).fillna(-np.inf).max(axis = 1).to_numpy()

def conceptCounts(generated : pd.DataFrame, gold : pd.DataFrame, concepts : list,
        match : str, viewSources : list, threshold : float | None = None) -> pd.DataFrame:
    """
    Per scored concept: its gold synonyms, the gold synonyms matched, its
    generated synonyms of the view and those matching a gold synonym. With
    `threshold` the embedding match uses it instead of
    syngen.evaluation.embedding_match.
    """
    if match == embeddingMatch:
        limit = (evaluationConfig.embedding_match if threshold is None else threshold) - similarityTolerance
        goldMatched = goldSimilarity(gold, viewSources) >= limit
        generatedMatched = generated[goldSimilarityColumn].astype(float).fillna(-np.inf).to_numpy() >= limit
    else:
        goldKeys      = keysOf(gold[synonymColumn], gold[normalizedColumn], gold[embeddingKeyColumn], match)
        generatedKeys = keysOf(generated[synonymColumn], generated[normalizedColumn],
            generated[embeddingKeyColumn], match)
        goldSets      = pd.Series(goldKeys).groupby(gold[idColumn].to_numpy()).agg(set).to_dict()
        generatedSets = pd.Series(generatedKeys).groupby(generated[idColumn].to_numpy()).agg(set).to_dict() \
            if len(generated.index) > 0 else {}
        goldMatched = np.array([key != "" and key in generatedSets.get(hpoID, set())
            for hpoID, key in zip(gold[idColumn], goldKeys)], dtype = bool)
        generatedMatched = np.array([key != "" and key in goldSets.get(hpoID, set())
            for hpoID, key in zip(generated[idColumn], generatedKeys)], dtype = bool)

    counts = pd.DataFrame(index = pd.Index(concepts, name = idColumn))
    counts["gold"]              = gold.groupby(idColumn).size()
    counts["gold_matched"]      = pd.Series(goldMatched, index = gold.index).groupby(gold[idColumn]).sum()
    if len(generated.index) > 0:
        counts["generated"]         = generated.groupby(idColumn).size()
        counts["generated_matched"] = pd.Series(generatedMatched, index = generated.index) \
            .groupby(generated[idColumn]).sum()
    else:
        counts["generated"], counts["generated_matched"] = 0, 0
    return counts.fillna(0).astype(int)

def scores(counts : pd.DataFrame) -> dict:
    """
    Micro and macro precision, recall and F1 of the per-concept counts.
    """
    def divide(a, b):
        return a / b if b > 0 else float("nan")
    def f1(p, r):
        return 2 * p * r / (p + r) if p + r > 0 else (0.0 if p == p and r == r else float("nan"))

    precision = divide(counts["generated_matched"].sum(), counts["generated"].sum())
    recall    = divide(counts["gold_matched"].sum(), counts["gold"].sum())

    withGenerated = counts[counts["generated"] > 0]
    conceptPrecision = withGenerated["generated_matched"] / withGenerated["generated"]
    conceptRecall    = counts["gold_matched"] / counts["gold"]
    conceptF1 = [f1(p, r) for p, r in zip(
        (counts["generated_matched"] / counts["generated"].replace(0, np.nan)).fillna(0), conceptRecall)]

    return {
        "precision"       : precision,
        "recall"          : recall,
        "f1"              : f1(precision, recall),
        "macro_precision" : conceptPrecision.mean() if len(conceptPrecision) > 0 else float("nan"),
        "macro_recall"    : conceptRecall.mean(),
        "macro_f1"        : float(np.mean(conceptF1)) if len(conceptF1) > 0 else float("nan")
    }

def bootstrap(counts : pd.DataFrame, samples : int, seed : int = 0) -> dict:
    """
    The 95% intervals of the micro precision, recall and F1, from samples of
    the concepts drawn with replacement.
    """
    generator = np.random.default_rng(seed)
    values = counts[["gold", "gold_matched", "generated", "generated_matched"]].to_numpy()
    intervals = {"precision" : [], "recall" : [], "f1" : []}
    for start in range(0, samples, 100):
        n = min(100, samples - start)
        drawn = generator.integers(0, len(values), size = (n, len(values)))
        sums  = values[drawn].sum(axis = 1)
        with np.errstate(divide = "ignore", invalid = "ignore"):
            precision = sums[:, 3] / sums[:, 2]
            recall    = sums[:, 1] / sums[:, 0]
            f1        = 2 * precision * recall / (precision + recall)
        intervals["precision"].extend(precision)
        intervals["recall"].extend(recall)
        intervals["f1"].extend(f1)
    ret = {}
    for name, sampled in intervals.items():
        sampled = np.asarray(sampled, dtype = float)
        sampled = sampled[~np.isnan(sampled)]
        low, high = (np.percentile(sampled, [2.5, 97.5]) if len(sampled) > 0 else (np.nan, np.nan))
        ret[f"{name}_low"], ret[f"{name}_high"] = low, high
    return ret

def accuracy(generated : pd.DataFrame, gold : pd.DataFrame, match : str) -> float:
    """
    The gold synonyms, deduplicated over all concepts by their key, whose key
    is that of any generated synonym of the view, of whatever concept; NaN
    for an embedding match below 1, which has no key.
    """
    if match == embeddingMatch and evaluationConfig.embedding_match < 1 - similarityTolerance:
        return float("nan")
    goldKeys = set(keysOf(gold[synonymColumn], gold[normalizedColumn], gold[embeddingKeyColumn], match)) - {""}
    generatedKeys = set(keysOf(generated[synonymColumn], generated[normalizedColumn],
        generated[embeddingKeyColumn], match))
    return len(goldKeys & generatedKeys) / len(goldKeys) if len(goldKeys) > 0 else float("nan")

def wordsCovered(generated : pd.DataFrame, gold : pd.DataFrame, labelWords : dict,
        concepts : list) -> dict:
    """
    The mean over the concepts of the Jaccard index of the words of the gold
    and of the generated synonyms (their normalised tokens), and of the share
    of gold words generated; also without the words of the label.
    """
    def words(frame : pd.DataFrame) -> dict:
        tokens = frame[normalizedColumn].fillna("").astype(str).str.split()
        return tokens.groupby(frame[idColumn].to_numpy()).agg(
            lambda lists: set(t for ts in lists for t in ts)).to_dict() if len(frame.index) > 0 else {}

    goldWords, generatedWords = words(gold), words(generated)
    results = {"words_jaccard" : [], "words_covered" : [], "new_words_jaccard" : [], "new_words_covered" : []}
    for hpoID in concepts:
        g, p = goldWords.get(hpoID, set()), generatedWords.get(hpoID, set())
        for prefix, (a, b) in [("words", (g, p)),
                ("new_words", (g - labelWords.get(hpoID, set()), p - labelWords.get(hpoID, set())))]:
            if len(a) == 0:
                continue
            results[f"{prefix}_jaccard"].append(len(a & b) / len(a | b))
            results[f"{prefix}_covered"].append(len(a & b) / len(a))
    return {name : float(np.mean(values)) if len(values) > 0 else float("nan")
        for name, values in results.items()}

def otherMatches(generated : pd.DataFrame, other : pd.DataFrame, gold : pd.DataFrame,
        labels : pd.DataFrame) -> dict:
    """
    How many generated synonyms (normalised) are not gold synonyms of their
    concept but synonyms HPO lists for it with another class (or excludes
    from the gold standard), and how many are a gold synonym or the label of
    another concept.
    """
    keys = list(zip(generated[idColumn], generated[normalizedColumn].fillna("").astype(str)))
    goldOwn = set(zip(gold[idColumn], gold[normalizedColumn].fillna("").astype(str)))

    ret = {}
    otherOwn = {}
    for hpoID, key, classes in zip(other[idColumn], other[normalizedColumn], other[sourceColumn]):
        otherOwn.setdefault((hpoID, key), classes.split(", ")[0] or "other")
    for key in keys:
        if key[1] != "" and key not in goldOwn and key in otherOwn:
            name = f"hpo_{otherOwn[key]}"
            ret[name] = ret.get(name, 0) + 1

    byKey = {}
    for hpoID, key in list(zip(gold[idColumn], gold[normalizedColumn].fillna("").astype(str))) + \
            list(zip(labels[idColumn], labels[normalizedColumn])):
        byKey.setdefault(key, set()).add(hpoID)
    ret["other_concept"] = sum(1 for hpoID, key in keys
        if key != "" and (hpoID, key) not in goldOwn and len(byKey.get(key, set()) - {hpoID}) > 0)
    return ret

# ==============================================================================
# Plots
# ==============================================================================

def plotSummary(summary : pd.DataFrame, file : str) -> None:
    """Precision, recall and F1 (micro) of every view and match."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    views = summary["view"].drop_duplicates().tolist()
    figure, axes = plt.subplots(1, len(matchTypes), figsize = (5 * len(matchTypes), 4), sharey = True)
    for axis, match in zip(np.atleast_1d(axes), matchTypes):
        rows = summary[summary["match"] == match].set_index("view").reindex(views)
        x = np.arange(len(views))
        for offset, metric in zip([-0.25, 0, 0.25], ["precision", "recall", "f1"]):
            axis.bar(x + offset, rows[metric], width = 0.25, label = metric)
        axis.set_xticks(x, views, rotation = 45, ha = "right")
        axis.set_title(match)
        axis.set_ylim(0, 1)
    np.atleast_1d(axes)[0].legend()
    figure.tight_layout()
    os.makedirs(os.path.dirname(file), exist_ok = True)
    figure.savefig(file, dpi = 150)
    plt.close(figure)

def plotCurve(curve : pd.DataFrame, x : str, file : str, xlabel : str) -> None:
    """Precision, recall and F1 (micro) of every view against x."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize = (15, 4), sharey = True)
    for axis, metric in zip(axes, ["precision", "recall", "f1"]):
        for view, rows in curve.groupby("view", sort = False):
            axis.plot(rows[x], rows[metric], label = view)
        axis.set_title(metric)
        axis.set_xlabel(xlabel)
        axis.set_ylim(0, 1)
    axes[0].legend()
    figure.tight_layout()
    os.makedirs(os.path.dirname(file), exist_ok = True)
    figure.savefig(file, dpi = 150)
    plt.close(figure)

# ==============================================================================
# The evaluation of one model's run.
# ==============================================================================

def evaluateModel(modelName : str) -> pd.DataFrame:
    """
    Evaluates the synonyms of one model's run and writes the summary, the
    curves and their plots.
    """
    exitIfFileNotExist(outputFileEmbedded(modelName))
    generatedAll = readCSV(outputFileEmbedded(modelName))
    generatedAll[synonymColumn] = generatedAll[synonymColumn].fillna("").astype(str)
    generatedAll[asLabelColumn] = generatedAll[asLabelColumn].astype(str).str.lower() == "true"
    gold = readCSV(outputFileEmbeddedGold(modelName))
    gold[synonymColumn] = gold[synonymColumn].fillna("").astype(str)

    data     = readTask()
    concepts = sorted(gold[idColumn].unique())
    l.log(f"Scoring {len(concepts)} concepts with {len(gold.index)} gold synonyms.")

    labelFrame = pd.DataFrame(list(labelsByConcept(data).items()), columns = [idColumn, valueColumn])
    labelFrame[normalizedColumn] = normalizedKeys(labelFrame[valueColumn].tolist())
    labelWords = {hpoID : set(key.split()) for hpoID, key in zip(labelFrame[idColumn], labelFrame[normalizedColumn])}
    other = otherSynonyms(data)
    other[normalizedColumn] = normalizedKeys(other[synonymColumn].tolist())

    rows, thresholdCurve, roundsCurve = [], [], []
    thresholds = np.round(np.arange(evaluationConfig.threshold_lower,
        evaluationConfig.threshold_upper + evaluationConfig.threshold_step / 2,
        evaluationConfig.threshold_step), 6)

    for stage in experimentStages:
        for mode in [aloneMode, cumulativeMode]:
            if mode == cumulativeMode and stage == experimentStages[0]:
                continue
            sources = viewStages(stage, mode)
            view    = stage if mode == aloneMode else f"{stage} ({mode})"
            ofView  = generatedAll[generatedAll[sourceColumn].isin(sources)]
            asLabel = int(ofView[asLabelColumn].sum())
            ofView  = ofView[~ofView[asLabelColumn]]
            # Only the synonyms of scored concepts count for precision.
            scored  = ofView[ofView[idColumn].isin(concepts)]

            words = wordsCovered(scored, gold, labelWords, concepts)
            others = otherMatches(scored, other, gold, labelFrame)
            for match in matchTypes:
                counts = conceptCounts(scored, gold, concepts, match, sources)
                row = {"view" : view, "stage" : stage, "mode" : mode, "match" : match,
                    "concepts" : len(concepts), "gold" : int(counts["gold"].sum()),
                    "generated" : int(counts["generated"].sum()),
                    "generated_unscored" : len(ofView.index) - len(scored.index),
                    "as_label" : asLabel,
                    "gold_matched" : int(counts["gold_matched"].sum()),
                    "generated_matched" : int(counts["generated_matched"].sum())}
                row |= scores(counts)
                row |= bootstrap(counts, evaluationConfig.bootstrap_samples)
                row["accuracy"] = accuracy(ofView, gold, match)
                row |= words
                row |= others
                rows.append(row)
                l.log(f"{view}, {match}: P {row['precision']:.3f}, R {row['recall']:.3f}, " \
                    f"F1 {row['f1']:.3f} [{row['f1_low']:.3f}, {row['f1_high']:.3f}], " \
                    f"accuracy {row['accuracy']:.3f}.")

            for threshold in thresholds:
                counts = conceptCounts(scored, gold, concepts, embeddingMatch, sources, threshold)
                thresholdCurve.append({"view" : view, "threshold" : threshold} | scores(counts))

            # The LLM synonyms generated in at least k rounds.
            if stage == llmStage and mode == aloneMode:
                for k in range(1, rounds + 1):
                    subset = scored[scored[roundsColumn].astype(float) >= k]
                    counts = conceptCounts(subset, gold, concepts, normalizedMatch, sources)
                    roundsCurve.append({"view" : view, "min_rounds" : k} | scores(counts))

            l.log(f"{view}: words covered {words['words_covered']:.3f} (Jaccard " \
                f"{words['words_jaccard']:.3f}), without the label's {words['new_words_covered']:.3f}; " \
                f"as label {asLabel}; " + ", ".join(f"{k} {v}" for k, v in others.items()) + ".")

    summary = pd.DataFrame(rows)
    writeHugeCSV(summary, outputFileEvaluation(modelName, "summary", csvFileFormat))
    thresholdCurve = pd.DataFrame(thresholdCurve)
    writeHugeCSV(thresholdCurve, outputFileEvaluation(modelName, "threshold_curve", csvFileFormat))
    roundsCurve = pd.DataFrame(roundsCurve)
    writeHugeCSV(roundsCurve, outputFileEvaluation(modelName, "rounds_curve", csvFileFormat))

    plotSummary(summary, outputFileEvaluation(modelName, "summary"))
    plotCurve(thresholdCurve, "threshold", outputFileEvaluation(modelName, "threshold_curve"),
        "cosine similarity threshold")
    if len(roundsCurve.index) > 0:
        plotCurve(roundsCurve, "min_rounds", outputFileEvaluation(modelName, "rounds_curve"),
            "generated in at least k rounds")

    final = summary[(summary["stage"] == experimentStages[-1]) &
        (summary["mode"] == (cumulativeMode if len(experimentStages) > 1 else aloneMode))]
    l.log("Final (cumulative) evaluation:")
    for _, row in final.iterrows():
        l.log(f"  {row['match']}: precision {row['precision']:.3f}, recall {row['recall']:.3f}, " \
            f"F1 {row['f1']:.3f}, accuracy {row['accuracy']:.3f}.")
    best = thresholdCurve.loc[thresholdCurve.groupby("view")["f1"].idxmax().dropna()]
    for _, row in best.iterrows():
        l.log(f"Best embedding threshold of {row['view']}: {row['threshold']} (F1 {row['f1']:.3f}).")
    return summary

def main() -> None:
    """
    Evaluates the experiment for every model.
    """
    l.log(f"Command: python3 {shlex.join(sys.argv)}")
    if experimentIndex is None:
        sys.exit("Give the index of the experiment.")

    l.printHeader(f"Evaluating Experiment {experimentIndex}")
    failed = []
    for modelID in modelIDs:
        try:
            with l.step(f"Evaluating ({modelNameOf(modelID)})"):
                evaluateModel(modelNameOf(modelID))
        except (Exception, SystemExit):
            l.exception(f"Evaluating {quote(modelID)} failed.", cmdline = False)
            failed.append(modelID)

    if len(failed) > 0:
        l.error(f"{len(failed)} of {len(modelIDs)} model(s) failed: {applyFormat(failed)}.")
        sys.exit(1)


if __name__ == "__main__":
    main()
