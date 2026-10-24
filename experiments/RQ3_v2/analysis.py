"""Analysis helpers for RQ3 v2 learning curves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LearningCurveSummary:
    """Compact sample-efficiency summary of one complete learning curve."""

    normalized_log_aulc: float
    auc_at_minimum_size: float
    auc_at_maximum_size: float
    gain: float
    maximum_observed_auc: float
    size_at_95_percent_improvement: int


@dataclass(frozen=True)
class FixedEffectSlope:
    """Linear slope after removing stratum-specific means."""

    slope: float
    effect_per_10_percentage_points: float
    observations: int
    informative_strata: int
    centered_coverage_sd: float


def nested_balanced_subsets(
    episodes: pd.DataFrame,
    training_sizes: Sequence[int],
    generator: np.random.Generator,
) -> Dict[int, Tuple[str, ...]]:
    """Draw deterministic nested subsets with exact 50/50 outcome balance."""

    required = {"episode_id", "failure"}
    missing = required - set(episodes.columns)
    if missing:
        raise ValueError("episodes missing columns: " + ", ".join(sorted(missing)))
    if episodes["episode_id"].astype(str).duplicated().any():
        raise ValueError("episode IDs must be unique")
    sizes = tuple(sorted(int(value) for value in training_sizes))
    if not sizes or len(set(sizes)) != len(sizes):
        raise ValueError("training sizes must be non-empty and distinct")
    if any(value < 2 or value % 2 for value in sizes):
        raise ValueError("training sizes must be even integers of at least two")

    maximum_per_class = sizes[-1] // 2
    ordered: Dict[int, np.ndarray] = {}
    labels = pd.to_numeric(episodes["failure"], errors="raise").astype(int)
    if set(labels.unique()) != {0, 1}:
        raise ValueError("episodes must contain both binary outcome classes")
    for label in (0, 1):
        available = np.asarray(sorted(
            episodes.loc[labels == label, "episode_id"].astype(str)), dtype=object)
        if len(available) < maximum_per_class:
            raise ValueError(
                "training pool lacks {} trajectories for class {}".format(
                    maximum_per_class, label))
        ordered[label] = generator.permutation(available)[:maximum_per_class]

    result: Dict[int, Tuple[str, ...]] = {}
    for size in sizes:
        per_class = size // 2
        selected = np.concatenate([
            ordered[0][:per_class], ordered[1][:per_class]
        ])
        result[size] = tuple(sorted(str(value) for value in selected))
    for smaller, larger in zip(sizes, sizes[1:]):
        if not set(result[smaller]) < set(result[larger]):
            raise AssertionError("generated suites are not strictly nested")
    return result


def summarize_learning_curve(
    training_sizes: Iterable[int],
    roc_auc: Iterable[float],
    target_fraction: float = 0.95,
    chance_level: float = 0.5,
) -> LearningCurveSummary:
    """Summarize a curve using log2(size)-weighted normalized area."""

    sizes = np.asarray(list(training_sizes), dtype=np.int64)
    values = np.asarray(list(roc_auc), dtype=np.float64)
    if sizes.ndim != 1 or values.ndim != 1 or sizes.size != values.size or sizes.size < 2:
        raise ValueError("a learning curve requires at least two paired points")
    order = np.argsort(sizes, kind="mergesort")
    sizes = sizes[order]
    values = values[order]
    if len(np.unique(sizes)) != len(sizes) or np.any(sizes <= 0):
        raise ValueError("training sizes must be unique positive integers")
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("ROC-AUC values must be finite values in [0, 1]")
    if not 0.0 < target_fraction <= 1.0:
        raise ValueError("target_fraction must lie in (0, 1]")

    x_values = np.log2(sizes.astype(np.float64))
    widths = np.diff(x_values)
    area = float(np.sum(widths * (values[:-1] + values[1:]) * 0.5))
    normalized = area / float(x_values[-1] - x_values[0])
    maximum = float(values.max())
    improvement = maximum - float(chance_level)
    target = (
        float(chance_level) + float(target_fraction) * improvement
        if improvement > 0.0 else maximum
    )
    reached = np.flatnonzero(values >= target)
    size_at_target = int(sizes[int(reached[0])]) if reached.size else int(sizes[-1])
    return LearningCurveSummary(
        normalized_log_aulc=normalized,
        auc_at_minimum_size=float(values[0]),
        auc_at_maximum_size=float(values[-1]),
        gain=float(values[-1] - values[0]),
        maximum_observed_auc=maximum,
        size_at_95_percent_improvement=size_at_target,
    )


def fixed_effect_slope(
    frame: pd.DataFrame,
    coverage_column: str,
    outcome_column: str,
    strata_columns: Sequence[str],
) -> FixedEffectSlope:
    """Estimate a within-stratum coverage slope by demeaning both variables."""

    required = {coverage_column, outcome_column, *strata_columns}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("frame missing columns: " + ", ".join(sorted(missing)))
    if frame.empty:
        raise ValueError("frame must not be empty")
    values = frame[[coverage_column, outcome_column]].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("coverage and outcome values must be finite")

    grouped = frame.groupby(list(strata_columns), sort=False, dropna=False)
    centered_coverage = (
        frame[coverage_column].astype(float)
        - grouped[coverage_column].transform("mean").astype(float)
    ).to_numpy(dtype=np.float64)
    centered_outcome = (
        frame[outcome_column].astype(float)
        - grouped[outcome_column].transform("mean").astype(float)
    ).to_numpy(dtype=np.float64)
    denominator = float(np.dot(centered_coverage, centered_coverage))
    if denominator <= np.finfo(np.float64).eps:
        raise ValueError("coverage has no within-stratum variation")
    slope = float(np.dot(centered_coverage, centered_outcome) / denominator)
    informative = int(sum(
        float(group[coverage_column].max()) > float(group[coverage_column].min())
        for _, group in grouped
    ))
    return FixedEffectSlope(
        slope=slope,
        effect_per_10_percentage_points=0.10 * slope,
        observations=len(frame),
        informative_strata=informative,
        centered_coverage_sd=float(np.sqrt(np.mean(centered_coverage**2))),
    )
