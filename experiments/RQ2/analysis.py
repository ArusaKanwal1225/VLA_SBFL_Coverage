"""Matched statistical helpers for the RQ2 voxel-size experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from experiments.RQ1.analysis import average_ranks


@dataclass(frozen=True)
class MatchedBootstrapResult:
    """Percentile intervals indexed by voxel size and technique."""

    score_lower: np.ndarray
    score_upper: np.ndarray
    delta_lower: np.ndarray
    delta_upper: np.ndarray
    interaction_lower: np.ndarray
    interaction_upper: np.ndarray


def matched_size_bootstrap(
    fold_scores: np.ndarray,
    primary_size_index: int,
    reference_technique_index: int,
    iterations: int,
    seed: int,
) -> MatchedBootstrapResult:
    """Bootstrap blocks and runs using the same draw for all sizes/techniques.

    ``fold_scores`` has dimensions
    ``block x budget x run x voxel_size x technique``. Budgets are always kept
    equally weighted, matching the RQ1 primary-block policy.
    """

    values = np.asarray(fold_scores, dtype=np.float64)
    if values.ndim != 5 or min(values.shape) < 1 or not np.isfinite(values).all():
        raise ValueError("fold_scores must be a non-empty finite 5-D array")
    if iterations < 100:
        raise ValueError("at least 100 bootstrap iterations are required")
    blocks, budgets, folds, sizes, techniques = values.shape
    if not 0 <= primary_size_index < sizes:
        raise ValueError("primary_size_index is outside the voxel-size dimension")
    if not 0 <= reference_technique_index < techniques:
        raise ValueError(
            "reference_technique_index is outside the technique dimension")

    generator = np.random.default_rng(seed)
    draws = np.empty((iterations, sizes, techniques), dtype=np.float64)
    for iteration in range(iterations):
        block_indices = generator.integers(0, blocks, size=blocks)
        fold_indices = generator.integers(
            0, folds, size=(blocks, budgets, folds))
        sampled = np.empty(
            (blocks, budgets, folds, sizes, techniques), dtype=np.float64)
        for block_position, block_index in enumerate(block_indices):
            for budget_index in range(budgets):
                sampled[block_position, budget_index] = values[
                    block_index,
                    budget_index,
                    fold_indices[block_position, budget_index],
                ]
        draws[iteration] = sampled.mean(axis=(0, 1, 2))

    deltas = draws - draws[:, [primary_size_index], :]
    interactions = deltas - deltas[:, :, [reference_technique_index]]
    return MatchedBootstrapResult(
        score_lower=np.percentile(draws, 2.5, axis=0),
        score_upper=np.percentile(draws, 97.5, axis=0),
        delta_lower=np.percentile(deltas, 2.5, axis=0),
        delta_upper=np.percentile(deltas, 97.5, axis=0),
        interaction_lower=np.percentile(interactions, 2.5, axis=0),
        interaction_upper=np.percentile(interactions, 97.5, axis=0),
    )


def spearman_rank_correlation(
    first: Iterable[float],
    second: Iterable[float],
) -> float:
    """Return a tie-aware Spearman correlation for two finite vectors."""

    left = np.asarray(list(first), dtype=np.float64)
    right = np.asarray(list(second), dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or left.size < 2:
        raise ValueError("rank-correlation inputs must be equal vectors of length >= 2")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("rank-correlation inputs must be finite")
    left_ranks = average_ranks(left)
    right_ranks = average_ranks(right)
    left_centered = left_ranks - left_ranks.mean()
    right_centered = right_ranks - right_ranks.mean()
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denominator == 0.0:
        return 1.0 if np.array_equal(left_ranks, right_ranks) else 0.0
    return float(np.dot(left_centered, right_centered) / denominator)
