"""Coverage selection and matched-analysis helpers for RQ3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from vla_sbfl.voxelizer import Voxel


@dataclass(frozen=True)
class CoverageStatistics:
    """Coverage of one voxel support against a reference trajectory set."""

    unique_fraction: float
    mean_known_fraction: float
    failure_known_fraction: float
    pass_known_fraction: float
    all_unknown_trajectories: int
    reference_voxels: int
    reference_trajectories: int


@dataclass(frozen=True)
class MatchedCoverageBootstrapResult:
    """Intervals for high-minus-low effects and technique interactions.

    Size-specific arrays are indexed by training size and technique. Overall
    arrays average the training sizes equally before computing percentiles.
    """

    effect_lower: np.ndarray
    effect_upper: np.ndarray
    overall_effect_lower: np.ndarray
    overall_effect_upper: np.ndarray
    overall_interaction_lower: np.ndarray
    overall_interaction_upper: np.ndarray


def voxel_union(visits: Mapping[str, Sequence[Voxel]]) -> set[Voxel]:
    """Return the union of binary trajectory coverage."""

    support: set[Voxel] = set()
    for covered in visits.values():
        support.update(tuple(int(value) for value in voxel) for voxel in covered)
    return support


def coverage_statistics(
    support: Iterable[Voxel],
    reference_visits: Mapping[str, Sequence[Voxel]],
    labels: Mapping[str, int] | None = None,
) -> CoverageStatistics:
    """Measure unique and trajectory-weighted coverage without using outcomes.

    Labels are used only to report class-conditional diagnostics; they never
    affect subset selection.
    """

    known = {tuple(int(value) for value in voxel) for voxel in support}
    if not reference_visits:
        raise ValueError("reference_visits must not be empty")
    if labels is not None and set(labels) != set(reference_visits):
        raise ValueError("labels and reference_visits must have identical IDs")

    reference_support = voxel_union(reference_visits)
    if not reference_support:
        raise ValueError("reference trajectories must visit at least one voxel")

    fractions = []
    failure_fractions = []
    pass_fractions = []
    all_unknown = 0
    for episode_id, covered in reference_visits.items():
        episode_support = {tuple(voxel) for voxel in covered}
        if not episode_support:
            raise ValueError("reference trajectory has empty coverage: " + episode_id)
        fraction = len(known & episode_support) / len(episode_support)
        fractions.append(fraction)
        all_unknown += int(fraction == 0.0)
        if labels is not None:
            target = failure_fractions if int(labels[episode_id]) == 1 else pass_fractions
            target.append(fraction)

    def optional_mean(values: Sequence[float]) -> float:
        return float(np.mean(values)) if values else float("nan")

    return CoverageStatistics(
        unique_fraction=len(known & reference_support) / len(reference_support),
        mean_known_fraction=float(np.mean(fractions)),
        failure_known_fraction=optional_mean(failure_fractions),
        pass_known_fraction=optional_mean(pass_fractions),
        all_unknown_trajectories=all_unknown,
        reference_voxels=len(reference_support),
        reference_trajectories=len(reference_visits),
    )


def select_coverage_quantiles(
    candidates: pd.DataFrame,
    targets: Mapping[str, float],
    metric: str = "calibration_mean_known_fraction",
) -> pd.DataFrame:
    """Select distinct candidate subsets nearest requested coverage quantiles."""

    required = {"candidate_index", "subset_sha256", metric}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError("candidate table missing columns: " + ", ".join(sorted(missing)))
    if candidates.empty:
        raise ValueError("candidate table must not be empty")
    if candidates["subset_sha256"].duplicated().any():
        raise ValueError("candidate subsets must be unique")
    values = candidates[metric].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("candidate coverage must be finite")
    if len(candidates) < len(targets):
        raise ValueError("not enough candidates for distinct coverage targets")

    selected_rows = []
    used: set[str] = set()
    for tier, quantile in targets.items():
        if not 0.0 <= float(quantile) <= 1.0:
            raise ValueError("coverage quantiles must lie in [0, 1]")
        target_value = float(np.quantile(values, float(quantile)))
        ranked = candidates.assign(
            _distance=np.abs(values - target_value)
        ).sort_values(["_distance", "candidate_index"], kind="mergesort")
        available = ranked[~ranked["subset_sha256"].astype(str).isin(used)]
        if available.empty:
            raise ValueError("could not select distinct coverage candidates")
        row = available.iloc[0].drop(labels="_distance").to_dict()
        row.update({
            "coverage_tier": str(tier),
            "target_quantile": float(quantile),
            "target_coverage": target_value,
        })
        used.add(str(row["subset_sha256"]))
        selected_rows.append(row)
    return pd.DataFrame(selected_rows)


def matched_high_low_effects(
    metrics: pd.DataFrame,
    low_tier: str = "low",
    high_tier: str = "high",
) -> pd.DataFrame:
    """Create paired high-minus-low coverage effects for every formula cell."""

    keys = [
        "voxel_size", "configuration_id", "vla", "condition", "budget",
        "test_run", "training_size", "selection_replicate", "formula_id",
        "technique",
    ]
    values = [
        "roc_auc", "average_precision", "calibration_mean_known_fraction",
        "test_mean_known_fraction", "test_unique_coverage",
    ]
    required = set(keys + values + ["coverage_tier"])
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError("metrics missing columns: " + ", ".join(sorted(missing)))
    selected = metrics[metrics["coverage_tier"].isin([low_tier, high_tier])]
    if selected.duplicated(keys + ["coverage_tier"]).any():
        raise ValueError("coverage metrics contain duplicate paired cells")
    pivot = selected.pivot(index=keys, columns="coverage_tier", values=values)
    required_columns = [(value, tier) for value in values for tier in (low_tier, high_tier)]
    if any(column not in pivot.columns for column in required_columns):
        raise ValueError("one or more high/low coverage pairs are incomplete")
    pivot = pivot.dropna(subset=required_columns)
    rows = pivot.index.to_frame(index=False)
    for value in values:
        rows[value + "_low"] = pivot[(value, low_tier)].to_numpy(dtype=np.float64)
        rows[value + "_high"] = pivot[(value, high_tier)].to_numpy(dtype=np.float64)
        rows["delta_" + value] = rows[value + "_high"] - rows[value + "_low"]
    return rows


def matched_coverage_bootstrap(
    fold_deltas: np.ndarray,
    reference_technique_index: int,
    iterations: int,
    seed: int,
) -> MatchedCoverageBootstrapResult:
    """Bootstrap matched coverage effects while preserving equal budget weights.

    The input has dimensions block x budget x run x training_size x technique.
    The same sampled blocks and runs are used for all training sizes and
    techniques, preserving every paired contrast.
    """

    values = np.asarray(fold_deltas, dtype=np.float64)
    if values.ndim != 5 or min(values.shape) < 1 or not np.isfinite(values).all():
        raise ValueError("fold_deltas must be a non-empty finite 5-D array")
    if iterations < 100:
        raise ValueError("at least 100 bootstrap iterations are required")
    blocks, budgets, folds, training_sizes, techniques = values.shape
    if not 0 <= reference_technique_index < techniques:
        raise ValueError(
            "reference_technique_index is outside the technique dimension")

    generator = np.random.default_rng(seed)
    draws = np.empty(
        (iterations, training_sizes, techniques), dtype=np.float64)
    for iteration in range(iterations):
        block_indices = generator.integers(0, blocks, size=blocks)
        fold_indices = generator.integers(
            0, folds, size=(blocks, budgets, folds))
        sampled = np.empty_like(values)
        for block_position, block_index in enumerate(block_indices):
            for budget_index in range(budgets):
                sampled[block_position, budget_index] = values[
                    block_index,
                    budget_index,
                    fold_indices[block_position, budget_index],
                ]
        draws[iteration] = sampled.mean(axis=(0, 1, 2))

    overall_draws = draws.mean(axis=1)
    interactions = (
        overall_draws
        - overall_draws[:, [reference_technique_index]]
    )
    return MatchedCoverageBootstrapResult(
        effect_lower=np.percentile(draws, 2.5, axis=0),
        effect_upper=np.percentile(draws, 97.5, axis=0),
        overall_effect_lower=np.percentile(overall_draws, 2.5, axis=0),
        overall_effect_upper=np.percentile(overall_draws, 97.5, axis=0),
        overall_interaction_lower=np.percentile(interactions, 2.5, axis=0),
        overall_interaction_upper=np.percentile(interactions, 97.5, axis=0),
    )
