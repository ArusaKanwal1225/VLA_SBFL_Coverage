"""Paired statistical helpers for the RQ2 baseline comparison."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np

try:
    from experiments.RQ1.analysis import (
        FriedmanResult,
        WilcoxonResult,
        friedman_test,
        holm_adjust,
        wilcoxon_signed_rank,
    )
except ModuleNotFoundError:  # Direct execution places this directory on sys.path.
    from experiments.RQ1.analysis import (  # type: ignore
        FriedmanResult,
        WilcoxonResult,
        friedman_test,
        holm_adjust,
        wilcoxon_signed_rank,
    )


def paired_hierarchical_bootstrap(
    reference: np.ndarray,
    candidate: np.ndarray,
    iterations: int,
    seed: int,
) -> Tuple[float, float, float, float]:
    """Bootstrap the candidate-minus-reference gap over blocks, budgets, and folds.

    Arrays have shape ``(primary_blocks, budgets, folds)``. Blocks and folds are
    resampled; both budgets remain equally weighted, matching the RQ1 analysis.
    The return value is ``(reference_low, reference_high, gap_low, gap_high)``.
    """

    left = np.asarray(reference, dtype=np.float64)
    right = np.asarray(candidate, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 3 or min(left.shape) < 1:
        raise ValueError("paired bootstrap arrays must have equal non-empty 3-D shape")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("paired bootstrap arrays must contain only finite values")
    if iterations < 100:
        raise ValueError("at least 100 bootstrap iterations are required")

    blocks, budgets, folds = left.shape
    generator = np.random.default_rng(seed)
    reference_draws = np.empty(iterations, dtype=np.float64)
    gap_draws = np.empty(iterations, dtype=np.float64)
    budget_indices = np.arange(budgets)[None, :, None]
    for index in range(iterations):
        block_indices = generator.integers(0, blocks, blocks)
        fold_indices = generator.integers(0, folds, size=(blocks, budgets, folds))
        sampled_reference = left[
            block_indices[:, None, None], budget_indices, fold_indices
        ]
        sampled_candidate = right[
            block_indices[:, None, None], budget_indices, fold_indices
        ]
        reference_draws[index] = sampled_reference.mean()
        gap_draws[index] = (sampled_candidate - sampled_reference).mean()
    return (
        float(np.percentile(reference_draws, 2.5)),
        float(np.percentile(reference_draws, 97.5)),
        float(np.percentile(gap_draws, 2.5)),
        float(np.percentile(gap_draws, 97.5)),
    )


__all__ = [
    "FriedmanResult",
    "WilcoxonResult",
    "friedman_test",
    "holm_adjust",
    "paired_hierarchical_bootstrap",
    "wilcoxon_signed_rank",
]
