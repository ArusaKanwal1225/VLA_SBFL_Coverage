"""Dependency-light paired statistical analysis for RQ1."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np


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


def wilcoxon_signed_rank(first: Iterable[float], second: Iterable[float]) -> WilcoxonResult:
    """Two-sided paired Wilcoxon signed-rank test with tie correction."""

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
    """Bootstrap blocks and folds while preserving equal budget weights."""

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
