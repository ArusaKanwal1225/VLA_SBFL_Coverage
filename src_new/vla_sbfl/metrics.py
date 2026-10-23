"""Dependency-light binary ranking metrics for held-out evaluation."""

from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def _validate_binary(y_true: Iterable[int], scores: Iterable[float]):
    labels = np.asarray(list(y_true), dtype=np.int8)
    values = np.asarray(list(scores), dtype=np.float64)
    if labels.ndim != 1 or values.ndim != 1 or len(labels) != len(values) or len(labels) == 0:
        raise ValueError("labels and scores must be non-empty one-dimensional arrays of equal length")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("ROC-AUC and AP require both binary classes")
    if not np.isfinite(values).all():
        raise ValueError("metric scores must be finite")
    return labels, values


def roc_auc(y_true: Iterable[int], scores: Iterable[float]) -> float:
    """Tie-aware ROC-AUC via the Mann-Whitney rank statistic."""

    labels, values = _validate_binary(y_true, scores)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start+1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        average_one_based_rank = 0.5*((start+1)+end)
        ranks[order[start:end]] = average_one_based_rank
        start = end
    positives = labels == 1
    n_positive = int(positives.sum())
    n_negative = len(labels)-n_positive
    statistic = ranks[positives].sum()-n_positive*(n_positive+1)/2
    return float(statistic/(n_positive*n_negative))


def average_precision(y_true: Iterable[int], scores: Iterable[float]) -> float:
    """Average precision using threshold groups, matching standard PR semantics."""

    labels, values = _validate_binary(y_true, scores)
    order = np.argsort(-values, kind="mergesort")
    labels = labels[order]
    values = values[order]
    total_positive = int(labels.sum())
    true_positive = 0
    false_positive = 0
    previous_recall = 0.0
    result = 0.0
    start = 0
    while start < len(values):
        end = start+1
        while end < len(values) and values[end] == values[start]:
            end += 1
        group_positive = int(labels[start:end].sum())
        true_positive += group_positive
        false_positive += (end-start)-group_positive
        recall = true_positive/total_positive
        precision = true_positive/(true_positive+false_positive)
        result += (recall-previous_recall)*precision
        previous_recall = recall
        start = end
    return float(result)


def evaluate_predictions(
    predictions: pd.DataFrame,
    formula_ids: Sequence[str],
    label_column: str = "failure",
) -> pd.DataFrame:
    if label_column not in predictions.columns:
        raise ValueError("predictions are missing label column " + label_column)
    labels = pd.to_numeric(predictions[label_column], errors="coerce")
    if labels.isna().any():
        raise ValueError("evaluation labels must be numeric")
    rows = []
    for formula_id in formula_ids:
        column = "score__"+formula_id
        if column not in predictions.columns:
            raise ValueError("predictions are missing score column " + column)
        values = predictions[column].to_numpy(dtype=np.float64)
        rows.append({
            "formula_id": formula_id,
            "roc_auc": roc_auc(labels, values),
            "average_precision": average_precision(labels, values),
            "episodes": len(labels),
            "failures": int(labels.sum()),
            "failure_prevalence": float(labels.mean()),
            "distinct_scores": int(np.unique(values).size),
        })
    return pd.DataFrame(rows).sort_values(
        ["roc_auc", "average_precision", "formula_id"],
        ascending=[False, False, True], ignore_index=True)

