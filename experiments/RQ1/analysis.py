"""Dependency-light paired statistical analysis for RQ1."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np


# Exhaustive sign enumeration grows as 2**n.  This cap allows the 16-block
# confirmatory analyses while bounding one exact calculation to 1,048,576
# sign assignments.  Callers must explicitly select the approximation above
# this limit; the exact method never falls back silently.
MAX_EXACT_NONZERO_PAIRS = 20


def average_ranks(values: Iterable[float], descending: bool = False) -> np.ndarray:
    """Return one-based, tie-aware average ranks."""

    data = np.asarray(list(values), dtype=np.float64)
    if data.ndim != 1 or data.size == 0 or not np.isfinite(data).all():
        raise ValueError("ranking values must be a non-empty finite vector")
    ordered_values = -data if descending else data
    order = np.argsort(ordered_values, kind="mergesort")
    result = np.empty(data.size, dtype=np.float64)
    start = 0
    while start < data.size:
        end = start + 1
        while end < data.size and ordered_values[order[end]] == ordered_values[order[start]]:
            end += 1
        result[order[start:end]] = 0.5 * ((start + 1) + end)
        start = end
    return result


def competition_ranks(values: Iterable[float], descending: bool = False) -> np.ndarray:
    """Return one-based competition ranks, with ties assigned the minimum rank."""

    data = np.asarray(list(values), dtype=np.float64)
    if data.ndim != 1 or data.size == 0 or not np.isfinite(data).all():
        raise ValueError("ranking values must be a non-empty finite vector")
    ordered_values = -data if descending else data
    order = np.argsort(ordered_values, kind="mergesort")
    result = np.empty(data.size, dtype=np.int64)
    start = 0
    while start < data.size:
        end = start + 1
        while end < data.size and ordered_values[order[end]] == ordered_values[order[start]]:
            end += 1
        result[order[start:end]] = start + 1
        start = end
    return result


def _regularized_gamma_q(shape: float, value: float) -> float:
    """Regularized upper incomplete gamma using stable series/continued fractions."""

    if shape <= 0 or value < 0 or not math.isfinite(shape) or not math.isfinite(value):
        raise ValueError("gamma arguments are outside the supported domain")
    if value == 0:
        return 1.0
    epsilon = 1e-14
    minimum = 1e-300
    iterations = 10_000
    log_scale = -value + shape * math.log(value) - math.lgamma(shape)
    if value < shape + 1.0:
        term = 1.0 / shape
        total = term
        denominator = shape
        for _ in range(iterations):
            denominator += 1.0
            term *= value / denominator
            total += term
            if abs(term) <= abs(total) * epsilon:
                lower = total * math.exp(log_scale)
                return min(1.0, max(0.0, 1.0 - lower))
        raise ArithmeticError("gamma series did not converge")

    b = value + 1.0 - shape
    c = 1.0 / minimum
    d = 1.0 / max(abs(b), minimum)
    if b < 0:
        d = -d
    fraction = d
    for index in range(1, iterations + 1):
        coefficient = -index * (index - shape)
        b += 2.0
        d = coefficient * d + b
        if abs(d) < minimum:
            d = minimum
        c = b + coefficient / c
        if abs(c) < minimum:
            c = minimum
        d = 1.0 / d
        delta = d * c
        fraction *= delta
        if abs(delta - 1.0) <= epsilon:
            upper = math.exp(log_scale) * fraction
            return min(1.0, max(0.0, upper))
    raise ArithmeticError("gamma continued fraction did not converge")


@dataclass(frozen=True)
class FriedmanResult:
    statistic: float
    degrees_of_freedom: int
    p_value: float
    blocks: int
    techniques: int


def friedman_test(score_matrix: np.ndarray) -> FriedmanResult:
    """Tie-corrected Friedman test; columns are paired techniques."""

    values = np.asarray(score_matrix, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 2 or not np.isfinite(values).all():
        raise ValueError("Friedman input must be a finite matrix with at least 2x2 values")
    blocks, techniques = values.shape
    ranks = np.vstack([average_ranks(row, descending=True) for row in values])
    rank_sums = ranks.sum(axis=0)
    statistic = (
        12.0 / (blocks * techniques * (techniques + 1.0))
        * float(np.square(rank_sums).sum())
        - 3.0 * blocks * (techniques + 1.0)
    )
    tie_sum = 0.0
    for row in values:
        _, counts = np.unique(row, return_counts=True)
        tie_sum += float(np.sum(counts**3 - counts))
    correction = 1.0 - tie_sum / (blocks * (techniques**3 - techniques))
    if correction <= 0:
        statistic = 0.0
        p_value = 1.0
    else:
        statistic /= correction
        p_value = _regularized_gamma_q(0.5 * (techniques - 1), 0.5 * statistic)
    return FriedmanResult(
        statistic=float(max(0.0, statistic)),
        degrees_of_freedom=techniques - 1,
        p_value=float(p_value),
        blocks=blocks,
        techniques=techniques,
    )


@dataclass(frozen=True)
class WilcoxonResult:
    statistic: float
    z_score: float
    p_value: float
    rank_biserial: float
    nonzero_pairs: int


def _exact_signed_rank_p_value(ranks: np.ndarray, positive: float) -> float:
    """Enumerate the conditional two-sided signed-rank distribution.

    The observed absolute differences, including their average ranks for ties,
    are held fixed.  Under the signed-rank null, every assignment of signs to
    the nonzero differences is equiprobable.  Ranks are doubled so that all
    subset sums and comparisons are performed exactly with integers.
    """

    count = int(ranks.size)
    if count > MAX_EXACT_NONZERO_PAIRS:
        raise ValueError(
            "exact Wilcoxon signed-rank enumeration supports at most {} "
            "nonzero pairs (received {}); select method='approximate' "
            "explicitly for a larger sample".format(
                MAX_EXACT_NONZERO_PAIRS, count))
    scaled_ranks = np.rint(2.0 * ranks).astype(np.int64)
    if not np.allclose(2.0 * ranks, scaled_ranks, rtol=0.0, atol=0.0):
        raise ArithmeticError("signed-rank averages are not integer or half-integer")
    positive_sum = int(round(2.0 * positive))
    total = int(scaled_ranks.sum())
    observed = min(positive_sum, total - positive_sum)

    subset_sums = np.zeros(1, dtype=np.int64)
    for rank in scaled_ranks:
        subset_sums = np.concatenate((subset_sums, subset_sums + rank))
    statistics = np.minimum(subset_sums, total - subset_sums)
    extreme = int(np.count_nonzero(statistics <= observed))
    return float(extreme / subset_sums.size)


def wilcoxon_signed_rank(
    first: Iterable[float],
    second: Iterable[float],
    method: str = "approximate",
) -> WilcoxonResult:
    """Two-sided paired Wilcoxon signed-rank test with tie correction.

    ``method="approximate"`` preserves the original normal approximation with
    tie and continuity corrections.  ``method="exact"`` conditions on the
    observed absolute nonzero differences and exhaustively enumerates their
    signs.  The exact test assumes that the paired differences are symmetric
    about zero under the null and independent signs (or joint invariance under
    all sign flips), making those sign assignments equiprobable. Marginal
    symmetry alone is insufficient when blocks are dependent; it
    is a test of the signed-rank location null, not specifically of equal
    arithmetic means.

    Exact enumeration is limited by :data:`MAX_EXACT_NONZERO_PAIRS`.  Exceeding
    the limit raises an error and never changes methods automatically.  The
    returned ``z_score`` remains the continuity-corrected standardized
    statistic in both modes for backward-compatible result structure; only the
    exact mode's ``p_value`` is obtained from enumeration.
    """

    if method not in {"approximate", "exact"}:
        raise ValueError("Wilcoxon method must be 'approximate' or 'exact'")

    left = np.asarray(list(first), dtype=np.float64)
    right = np.asarray(list(second), dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or left.size == 0:
        raise ValueError("paired samples must be non-empty vectors with equal shape")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("paired samples must be finite")
    differences = left - right
    differences = differences[differences != 0]
    count = differences.size
    if count == 0:
        return WilcoxonResult(0.0, 0.0, 1.0, 0.0, 0)
    absolute = np.abs(differences)
    ranks = average_ranks(absolute)
    positive = float(ranks[differences > 0].sum())
    negative = float(ranks[differences < 0].sum())
    statistic = min(positive, negative)
    mean = count * (count + 1.0) / 4.0
    _, ties = np.unique(absolute, return_counts=True)
    variance = (
        count * (count + 1.0) * (2.0 * count + 1.0)
        - float(np.sum(ties**3 - ties)) / 2.0
    ) / 24.0
    if variance <= 0:
        z_score = 0.0
        p_value = 1.0
    else:
        deviation = max(0.0, abs(positive - mean) - 0.5)
        z_score = deviation / math.sqrt(variance)
        p_value = math.erfc(z_score / math.sqrt(2.0))
    if method == "exact":
        p_value = _exact_signed_rank_p_value(ranks, positive)
    rank_total = count * (count + 1.0) / 2.0
    return WilcoxonResult(
        statistic=statistic,
        z_score=float(z_score),
        p_value=float(min(1.0, max(0.0, p_value))),
        rank_biserial=float((positive - negative) / rank_total),
        nonzero_pairs=int(count),
    )


def holm_adjust(p_values: Dict[str, float]) -> Dict[str, float]:
    """Holm step-down family-wise error correction."""

    if any(not math.isfinite(value) or value < 0 or value > 1 for value in p_values.values()):
        raise ValueError("p-values must be finite values in [0, 1]")
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: Dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, (key, value) in enumerate(ordered):
        running = max(running, (count - index) * value)
        adjusted[key] = min(1.0, running)
    return adjusted


@dataclass(frozen=True)
class PairwiseWilcoxonResult:
    first_id: str
    second_id: str
    statistic: float
    p_value: float
    holm_adjusted_p_value: float
    rank_biserial_first_minus_second: float
    nonzero_pairs: int


def all_pairwise_wilcoxon(
    score_matrix: np.ndarray,
    technique_ids: Sequence[str],
    method: str = "exact",
) -> Tuple[PairwiseWilcoxonResult, ...]:
    """Compare every technique pair and apply one Holm correction family.

    Rows of ``score_matrix`` are paired experimental blocks and columns follow
    ``technique_ids``.  The result contains exactly ``k * (k - 1) / 2``
    comparisons.  Holm adjustment is applied once to that complete family;
    selecting a presentation reference is deliberately left to the caller.
    ``z_score`` is not carried into these results because it does not determine
    exact p-values.
    """

    values = np.asarray(score_matrix, dtype=np.float64)
    labels = tuple(str(value) for value in technique_ids)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 2:
        raise ValueError("pairwise scores must be a matrix with at least two techniques")
    if values.shape[1] != len(labels):
        raise ValueError("technique_ids must match the score-matrix columns")
    if len(set(labels)) != len(labels):
        raise ValueError("technique_ids must be unique")
    if not np.isfinite(values).all():
        raise ValueError("pairwise scores must be finite")

    raw_p_values: Dict[str, float] = {}
    unadjusted = []
    for first_index, first_id in enumerate(labels[:-1]):
        for second_index in range(first_index + 1, len(labels)):
            second_id = labels[second_index]
            result = wilcoxon_signed_rank(
                values[:, first_index], values[:, second_index], method=method)
            key = "{}|{}".format(first_index, second_index)
            raw_p_values[key] = result.p_value
            unadjusted.append((key, first_id, second_id, result))

    adjusted = holm_adjust(raw_p_values)
    return tuple(
        PairwiseWilcoxonResult(
            first_id=first_id,
            second_id=second_id,
            statistic=result.statistic,
            p_value=result.p_value,
            holm_adjusted_p_value=adjusted[key],
            rank_biserial_first_minus_second=result.rank_biserial,
            nonzero_pairs=result.nonzero_pairs,
        )
        for key, first_id, second_id, result in unadjusted
    )


def hierarchical_bootstrap(
    fold_scores: np.ndarray,
    leader_index: int,
    iterations: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap configurations, then folds within sampled configurations."""

    values = np.asarray(fold_scores, dtype=np.float64)
    if values.ndim != 3 or min(values.shape) < 1 or not np.isfinite(values).all():
        raise ValueError("fold_scores must be a non-empty finite 3-D array")
    if iterations < 100:
        raise ValueError("at least 100 bootstrap iterations are required")
    configurations, folds, techniques = values.shape
    if leader_index < 0 or leader_index >= techniques:
        raise ValueError("leader_index is outside the technique dimension")
    generator = np.random.default_rng(seed)
    draws = np.empty((iterations, techniques), dtype=np.float64)
    for iteration in range(iterations):
        configuration_indices = generator.integers(0, configurations, configurations)
        fold_indices = generator.integers(0, folds, size=(configurations, folds))
        sample = values[configuration_indices[:, None], fold_indices, :]
        draws[iteration] = sample.mean(axis=(0, 1))
    lower = np.percentile(draws, 2.5, axis=0)
    upper = np.percentile(draws, 97.5, axis=0)
    deltas = draws[:, [leader_index]] - draws
    delta_lower = np.percentile(deltas, 2.5, axis=0)
    delta_upper = np.percentile(deltas, 97.5, axis=0)
    return lower, upper, delta_lower, delta_upper


def hierarchical_budget_bootstrap(
    fold_scores: np.ndarray,
    leader_index: int,
    iterations: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap paired 4-VLA x 4-strategy blocks and folds within fixed budgets.

    Percentile 95% intervals are pointwise, neither simultaneous nor adjusted
    for Holm multiplicity or selection of the observed reference technique.
    Resampling assumes adequate exchangeability of blocks and inner folds;
    shared factors and overlapping training folds limit this interpretation.
    """

    values = np.asarray(fold_scores, dtype=np.float64)
    if values.ndim != 4 or min(values.shape) < 1 or not np.isfinite(values).all():
        raise ValueError("fold_scores must be a non-empty finite 4-D array")
    if iterations < 100:
        raise ValueError("at least 100 bootstrap iterations are required")
    blocks, budgets, folds, techniques = values.shape
    if leader_index < 0 or leader_index >= techniques:
        raise ValueError("leader_index is outside the technique dimension")
    generator = np.random.default_rng(seed)
    draws = np.empty((iterations, techniques), dtype=np.float64)
    budget_indices = np.arange(budgets)[None, :, None]
    for iteration in range(iterations):
        block_indices = generator.integers(0, blocks, blocks)
        fold_indices = generator.integers(0, folds, size=(blocks, budgets, folds))
        sample = values[block_indices[:, None, None], budget_indices, fold_indices, :]
        draws[iteration] = sample.mean(axis=(0, 1, 2))
    lower = np.percentile(draws, 2.5, axis=0)
    upper = np.percentile(draws, 97.5, axis=0)
    deltas = draws[:, [leader_index]] - draws
    delta_lower = np.percentile(deltas, 2.5, axis=0)
    delta_upper = np.percentile(deltas, 97.5, axis=0)
    return lower, upper, delta_lower, delta_upper
