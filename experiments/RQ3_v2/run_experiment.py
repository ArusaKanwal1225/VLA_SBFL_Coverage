#!/usr/bin/env python3
"""Run RQ3 v2 nested trajectory-count learning curves."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from experiments.RQ1.analysis import (
    MAX_EXACT_NONZERO_PAIRS, holm_adjust, wilcoxon_signed_rank,
)
from experiments.RQ1.run_experiment import (
    EXPECTED_RUNS,
    Configuration,
    atomic_json,
    directory_size,
    discover_configurations,
    run_sort_key,
    sha256_file,
    subset_dataset,
    validate_configuration_metadata,
    validate_primary_block_metrics,
)
from experiments.RQ3.analysis import coverage_statistics, voxel_union
from experiments.RQ3_v2.analysis import (
    fixed_effect_slope,
    nested_balanced_subsets,
    summarize_learning_curve,
)
from vla_sbfl.dataset import TrajectoryDataset, load_trajectory_dataset
from vla_sbfl.formulas import FORMULA_REGISTRY, STUDY_FORMULA_IDS
from vla_sbfl.metrics import evaluate_predictions
from vla_sbfl.model import SBFLModel, episode_id_hash
from vla_sbfl.scoring import TrajectoryScorer
from vla_sbfl.voxelizer import VoxelGrid


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VOXEL_SIZES = (0.02,)
DEFAULT_TRAINING_SIZES = (10, 20, 50, 100, 200)
DEFAULT_REPETITIONS = 10
DEFAULT_SEED = 20261024
REFERENCE_FORMULA = "harmonic_mean"
PRACTICAL_MARGIN = 0.01


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def size_slug(voxel_size: float) -> str:
    return "size_" + format(float(voxel_size), ".2f").replace(".", "p")


def training_slug(training_size: int) -> str:
    return "train_{:04d}".format(int(training_size))


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = "\x1f".join([str(int(base_seed)), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def sha256_payload(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def choose_calibration_run(test_run: str, runs: Sequence[str]) -> str:
    ordered = tuple(sorted((str(value) for value in runs), key=run_sort_key))
    if test_run not in ordered or len(ordered) < 3:
        raise ValueError("test run must belong to a sequence of at least three runs")
    return ordered[(ordered.index(test_run) + 1) % len(ordered)]


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT,
            stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def experiment_signature(
    voxel_sizes: Sequence[float],
    origin: Sequence[float],
    training_sizes: Sequence[int],
    repetitions: int,
    sampling_seed: int,
    configuration_ids: Sequence[str],
    folds_per_configuration: int,
) -> Tuple[str, dict]:
    directory = Path(__file__).resolve().parent
    catalogue = [{
        "formula_id": formula_id,
        "name": FORMULA_REGISTRY[formula_id].name,
        "expression": FORMULA_REGISTRY[formula_id].expression,
    } for formula_id in STUDY_FORMULA_IDS]
    details = {
        "protocol_sha256": sha256_file(directory / "RQ3_v2.md"),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "analysis_sha256": sha256_file(directory / "analysis.py"),
        "voxel_sizes": [float(value) for value in voxel_sizes],
        "origin": [float(value) for value in origin],
        "coordinate_frame": "robot_base",
        "training_sizes": [int(value) for value in training_sizes],
        "repetitions": int(repetitions),
        "sampling_seed": int(sampling_seed),
        "configuration_ids": list(configuration_ids),
        "folds_per_configuration": int(folds_per_configuration),
        "class_balance": "exactly 50% pass and 50% fail",
        "sampling": "independent repetitions of nested class-specific permutations",
        "split": "one test run, next cyclic run calibration, remaining eight runs pool",
        "formula_catalogue": catalogue,
        "trajectory_aggregation": "maximum_training_voxel_empirical_percentile",
        "all_unknown_policy": "neutral_0.5",
        "primary_sample_efficiency_metric": "normalized area under ROC-AUC versus log2(n)",
    }
    return sha256_payload(details), details


def cell_directory(
    output_root: Path,
    voxel_size: float,
    configuration: Configuration,
    test_run: str,
    repetition: int,
    training_size: int,
) -> Path:
    return (
        output_root / "cells" / size_slug(voxel_size) / "folds"
        / configuration.configuration_id / test_run
        / "rep_{:02d}".format(repetition) / training_slug(training_size)
    )


def write_nested_suite_audit(
    destination: Path,
    pool: TrajectoryDataset,
    suites: Mapping[int, Sequence[str]],
) -> None:
    """Persist one compact audit row per trajectory in the largest suite."""

    destination.mkdir(parents=True, exist_ok=True)
    sizes = sorted(suites)
    largest = set(suites[sizes[-1]])
    rows = []
    labels = pool.episodes.set_index(
        pool.episodes["episode_id"].astype(str))["failure"].astype(int).to_dict()
    for episode_id in sorted(largest):
        first_size = min(size for size in sizes if episode_id in set(suites[size]))
        rows.append({
            "episode_id": episode_id,
            "failure": int(labels[episode_id]),
            "first_included_size": int(first_size),
        })
    pd.DataFrame(rows).to_csv(
        destination / "nested_suite_membership.csv.gz", index=False, compression="gzip")
    manifest = []
    previous: set[str] = set()
    for size in sizes:
        current = set(str(value) for value in suites[size])
        manifest.append({
            "training_size": int(size),
            "subset_sha256": episode_id_hash(tuple(sorted(current))),
            "training_failures": int(sum(labels[value] for value in current)),
            "training_passes": int(sum(1 - labels[value] for value in current)),
            "contains_previous_suite": bool(not previous or previous < current),
        })
        previous = current
    pd.DataFrame(manifest).to_csv(destination / "nested_suite_manifest.csv", index=False)


def run_learning_curve_cell(
    pool: TrajectoryDataset,
    calibration: TrajectoryDataset,
    testing: TrajectoryDataset,
    configuration: Configuration,
    test_run: str,
    calibration_run: str,
    grid: VoxelGrid,
    output_root: Path,
    signature: str,
    training_size: int,
    episode_ids: Sequence[str],
    repetition: int,
    resume: bool,
    save_models: bool,
    save_predictions: bool,
) -> dict:
    destination = cell_directory(
        output_root, grid.voxel_size, configuration, test_run,
        repetition, training_size)
    completion_path = destination / "cell.json"
    if completion_path.is_file():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("experiment_signature") != signature:
            raise ValueError("completed cell has a different signature: " + str(destination))
        if not resume:
            raise FileExistsError("completed cell exists and resume is disabled: " + str(destination))
        return completion

    training = subset_dataset(pool, episode_ids)
    training_ids = set(training.episodes["episode_id"].astype(str))
    calibration_ids = set(calibration.episodes["episode_id"].astype(str))
    testing_ids = set(testing.episodes["episode_id"].astype(str))
    if len(training.episodes) != training_size:
        raise ValueError("selected training suite has the wrong size")
    if int(training.episodes["failure"].sum()) != training_size // 2:
        raise ValueError("selected training suite is not exactly class-balanced")
    if training_ids & calibration_ids or training_ids & testing_ids or calibration_ids & testing_ids:
        raise ValueError("pool, calibration, and evaluation trajectories must be disjoint")

    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"episode_id": sorted(training_ids)}).to_csv(
        destination / "training_episode_ids.csv.gz", index=False, compression="gzip")
    subset_hash = episode_id_hash(tuple(sorted(training_ids)))
    support = voxel_union(training.visits)
    calibration_coverage = coverage_statistics(support, calibration.visits, calibration.labels)
    test_coverage = coverage_statistics(support, testing.visits, testing.labels)

    started = time.perf_counter()
    fit_started = time.perf_counter()
    model = SBFLModel.fit(
        visits=training.visits,
        labels=training.labels,
        grid=grid,
        formula_ids=STUDY_FORMULA_IDS,
        metadata={
            "experiment": "RQ3_v2_learning_curves",
            "experiment_signature": signature,
            "configuration_id": configuration.configuration_id,
            "vla": configuration.vla,
            "condition": configuration.condition,
            "budget": configuration.budget,
            "test_run": test_run,
            "calibration_run": calibration_run,
            "training_size": training_size,
            "sampling_repetition": repetition,
            "subset_sha256": subset_hash,
        },
    )
    fit_seconds = time.perf_counter() - fit_started
    model_artifact_bytes = 0
    if save_models:
        model.save(destination / "model")
        model_artifact_bytes = directory_size(destination / "model")

    score_started = time.perf_counter()
    predictions = TrajectoryScorer(model).score(testing.visits, testing.episodes)
    score_seconds = time.perf_counter() - score_started
    if save_predictions:
        predictions.to_csv(
            destination / "predictions.csv.gz", index=False, compression="gzip")

    metrics = evaluate_predictions(predictions, model.formula_ids)
    identifiers = {
        "voxel_size": float(grid.voxel_size),
        "configuration_id": configuration.configuration_id,
        "vla": configuration.vla,
        "condition": configuration.condition,
        "budget": configuration.budget,
        "test_run": test_run,
        "calibration_run": calibration_run,
        "training_size": int(training_size),
        "sampling_repetition": int(repetition),
        "subset_sha256": subset_hash,
    }
    for column, value in reversed(list(identifiers.items())):
        metrics.insert(0, column, value)
    metrics["technique"] = metrics["formula_id"].map(
        {formula_id: FORMULA_REGISTRY[formula_id].name for formula_id in model.formula_ids})
    metrics["training_failures"] = int(training.episodes["failure"].sum())
    metrics["training_passes"] = int((training.episodes["failure"] == 0).sum())
    metrics["training_voxels"] = len(model.voxels)
    metrics["calibration_unique_coverage"] = calibration_coverage.unique_fraction
    metrics["calibration_mean_known_fraction"] = calibration_coverage.mean_known_fraction
    metrics["test_unique_coverage"] = test_coverage.unique_fraction
    metrics["test_mean_known_fraction"] = test_coverage.mean_known_fraction
    metrics["test_failure_known_fraction"] = test_coverage.failure_known_fraction
    metrics["test_pass_known_fraction"] = test_coverage.pass_known_fraction
    metrics["all_unknown_trajectories"] = test_coverage.all_unknown_trajectories
    metrics["score_tie_fraction"] = 1.0 - metrics["distinct_scores"] / metrics["episodes"]
    metrics["fit_seconds"] = fit_seconds
    metrics["score_seconds"] = score_seconds
    metrics["model_artifact_bytes"] = model_artifact_bytes
    metrics.to_csv(destination / "metrics.csv", index=False)

    completion = {
        "status": "complete",
        "completed_at_utc": utc_now(),
        "experiment_signature": signature,
        **identifiers,
        "training_failures": int(training.episodes["failure"].sum()),
        "training_passes": int((training.episodes["failure"] == 0).sum()),
        "training_voxels": len(model.voxels),
        "test_episodes": len(testing.episodes),
        "test_failures": int(testing.episodes["failure"].sum()),
        "test_passes": int((testing.episodes["failure"] == 0).sum()),
        "calibration_unique_coverage": calibration_coverage.unique_fraction,
        "calibration_mean_known_fraction": calibration_coverage.mean_known_fraction,
        "test_unique_coverage": test_coverage.unique_fraction,
        "test_mean_known_fraction": test_coverage.mean_known_fraction,
        "test_failure_known_fraction": test_coverage.failure_known_fraction,
        "test_pass_known_fraction": test_coverage.pass_known_fraction,
        "all_unknown_trajectories": test_coverage.all_unknown_trajectories,
        "fit_seconds": fit_seconds,
        "score_seconds": score_seconds,
        "total_seconds": time.perf_counter() - started,
        "model_artifact_bytes": model_artifact_bytes,
    }
    atomic_json(completion_path, completion)
    return completion


MEASURE_COLUMNS = {
    "roc_auc": ("roc_auc", "mean"),
    "average_precision": ("average_precision", "mean"),
    "failure_prevalence": ("failure_prevalence", "mean"),
    "calibration_unique_coverage": ("calibration_unique_coverage", "mean"),
    "calibration_mean_known_fraction": ("calibration_mean_known_fraction", "mean"),
    "test_unique_coverage": ("test_unique_coverage", "mean"),
    "test_mean_known_fraction": ("test_mean_known_fraction", "mean"),
    "test_failure_known_fraction": ("test_failure_known_fraction", "mean"),
    "test_pass_known_fraction": ("test_pass_known_fraction", "mean"),
    "training_voxels": ("training_voxels", "mean"),
    "all_unknown_trajectories": ("all_unknown_trajectories", "mean"),
    "score_tie_fraction": ("score_tie_fraction", "mean"),
}


def learning_curve_units(metrics: pd.DataFrame, training_sizes: Sequence[int]) -> pd.DataFrame:
    """Compute paired curve summaries before any benchmark-level averaging."""

    keys = [
        "voxel_size", "configuration_id", "vla", "condition", "budget",
        "test_run", "sampling_repetition", "formula_id", "technique",
    ]
    expected = tuple(sorted(int(value) for value in training_sizes))
    rows: List[dict] = []
    for identity, frame in metrics.groupby(keys, sort=False):
        ordered = frame.sort_values("training_size", kind="mergesort")
        sizes = tuple(ordered["training_size"].astype(int))
        if sizes != expected:
            raise ValueError("one or more learning curves are incomplete")
        summary = summarize_learning_curve(sizes, ordered["roc_auc"])
        row = dict(zip(keys, identity))
        row.update({
            "normalized_log_aulc": summary.normalized_log_aulc,
            "auc_at_minimum_size": summary.auc_at_minimum_size,
            "auc_at_maximum_size": summary.auc_at_maximum_size,
            "gain_minimum_to_maximum": summary.gain,
            "maximum_observed_auc": summary.maximum_observed_auc,
            "size_at_95_percent_improvement": summary.size_at_95_percent_improvement,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_learning_curves(metrics: pd.DataFrame, aggregate: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    configuration = metrics.groupby([
        "voxel_size", "configuration_id", "vla", "condition", "budget",
        "training_size", "formula_id", "technique",
    ], as_index=False).agg(**MEASURE_COLUMNS)
    configuration.to_csv(aggregate / "configuration_learning_curves.csv", index=False)

    block = configuration.groupby([
        "voxel_size", "vla", "condition", "training_size", "formula_id", "technique",
    ], as_index=False).agg(**MEASURE_COLUMNS)
    block["primary_block_id"] = block["vla"].astype(str) + "__" + block["condition"].astype(str)
    block.to_csv(aggregate / "primary_block_learning_curves.csv", index=False)

    by_vla = block.groupby([
        "voxel_size", "vla", "training_size", "formula_id", "technique",
    ], as_index=False).agg(**MEASURE_COLUMNS)
    by_vla["roc_auc_rank"] = by_vla.groupby([
        "voxel_size", "vla", "training_size"
    ])["roc_auc"].rank(method="average", ascending=False)
    by_vla.to_csv(aggregate / "learning_curves_by_vla.csv", index=False)

    global_curves = block.groupby([
        "voxel_size", "training_size", "formula_id", "technique",
    ], as_index=False).agg(**MEASURE_COLUMNS)
    global_curves["roc_auc_rank"] = global_curves.groupby([
        "voxel_size", "training_size"
    ])["roc_auc"].rank(method="average", ascending=False)
    global_curves.to_csv(aggregate / "global_learning_curves.csv", index=False)
    return block, global_curves


def aggregate_sample_efficiency(
    unit_curves: pd.DataFrame,
    aggregate: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    measures = {
        "normalized_log_aulc": ("normalized_log_aulc", "mean"),
        "auc_at_minimum_size": ("auc_at_minimum_size", "mean"),
        "auc_at_maximum_size": ("auc_at_maximum_size", "mean"),
        "gain_minimum_to_maximum": ("gain_minimum_to_maximum", "mean"),
        "maximum_observed_auc": ("maximum_observed_auc", "mean"),
        "size_at_95_percent_improvement": ("size_at_95_percent_improvement", "mean"),
    }
    configuration = unit_curves.groupby([
        "voxel_size", "configuration_id", "vla", "condition", "budget",
        "formula_id", "technique",
    ], as_index=False).agg(**measures)
    block = configuration.groupby([
        "voxel_size", "vla", "condition", "formula_id", "technique",
    ], as_index=False).agg(**measures)
    block["primary_block_id"] = block["vla"].astype(str) + "__" + block["condition"].astype(str)
    block.to_csv(aggregate / "primary_block_sample_efficiency.csv", index=False)

    by_vla = block.groupby([
        "voxel_size", "vla", "formula_id", "technique",
    ], as_index=False).agg(**measures)
    by_vla["naulc_rank"] = by_vla.groupby([
        "voxel_size", "vla"
    ])["normalized_log_aulc"].rank(method="average", ascending=False)
    by_vla.to_csv(aggregate / "sample_efficiency_by_vla.csv", index=False)

    global_efficiency = block.groupby([
        "voxel_size", "formula_id", "technique",
    ], as_index=False).agg(**measures)
    global_efficiency["naulc_rank"] = global_efficiency.groupby(
        "voxel_size")["normalized_log_aulc"].rank(method="average", ascending=False)
    global_efficiency = global_efficiency.sort_values(
        ["voxel_size", "naulc_rank", "formula_id"], kind="mergesort")
    global_efficiency.to_csv(aggregate / "global_sample_efficiency.csv", index=False)
    return block, global_efficiency


def adjacent_reference_tests(
    block_curves: pd.DataFrame,
    training_sizes: Sequence[int],
    method: str = "approximate",
    include_counts: bool = False,
) -> Tuple[pd.DataFrame, dict]:
    primary = block_curves[
        np.isclose(block_curves["voxel_size"].astype(float), 0.02)
        & (block_curves["formula_id"].astype(str) == REFERENCE_FORMULA)
    ]
    pivot = primary.pivot(
        index="primary_block_id", columns="training_size", values="roc_auc")
    sizes = tuple(sorted(int(value) for value in training_sizes))
    if pivot.shape[0] != 16 or any(size not in pivot.columns for size in sizes):
        return pd.DataFrame(), {
            "status": "not_run",
            "reason": "complete 16-block reference learning curves are required",
        }
    raw: Dict[str, float] = {}
    tests = {}
    for lower, upper in zip(sizes, sizes[1:]):
        key = "{}-to-{}".format(lower, upper)
        result = wilcoxon_signed_rank(pivot[upper], pivot[lower], method=method)
        raw[key] = result.p_value
        tests[key] = result
    adjusted = holm_adjust(raw)
    rows = []
    for lower, upper in zip(sizes, sizes[1:]):
        key = "{}-to-{}".format(lower, upper)
        differences = pivot[upper] - pivot[lower]
        rows.append({
            "lower_training_size": lower,
            "upper_training_size": upper,
            "mean_delta_roc_auc": float(differences.mean()),
            "median_delta_roc_auc": float(differences.median()),
            "positive_block_fraction": float((differences > 0).mean()),
            "wilcoxon_p_value": tests[key].p_value,
            "holm_adjusted_p_value": adjusted[key],
            "rank_biserial": tests[key].rank_biserial,
            "primary_blocks": len(differences),
            **({"nonzero_pairs": tests[key].nonzero_pairs} if include_counts else {}),
        })
    return pd.DataFrame(rows), {
        "status": "complete",
        "reference": REFERENCE_FORMULA,
        "test": "paired Wilcoxon for four adjacent suite-size changes",
        "multiplicity": "Holm across four adjacent-size tests",
        "primary_blocks": 16,
    }


def sample_efficiency_tests(
    block_efficiency: pd.DataFrame, method: str = "approximate", include_counts: bool = False,
) -> Tuple[pd.DataFrame, dict]:
    primary = block_efficiency[
        np.isclose(block_efficiency["voxel_size"].astype(float), 0.02)]
    pivot = primary.pivot(
        index="primary_block_id", columns="formula_id", values="normalized_log_aulc")
    formula_ids = list(STUDY_FORMULA_IDS)
    if (
        pivot.shape[0] != 16
        or any(formula_id not in pivot.columns for formula_id in formula_ids)
        or pivot[formula_ids].isna().any().any()
    ):
        return pd.DataFrame(), {
            "status": "not_run",
            "reason": "complete 16-block sample-efficiency curves are required",
        }
    reference = pivot[REFERENCE_FORMULA].to_numpy(dtype=np.float64)
    raw: Dict[str, float] = {}
    tests = {}
    for formula_id in formula_ids:
        if formula_id == REFERENCE_FORMULA:
            continue
        result = wilcoxon_signed_rank(pivot[formula_id], reference, method=method)
        raw[formula_id] = result.p_value
        tests[formula_id] = result
    adjusted = holm_adjust(raw)
    rows = []
    for formula_id in formula_ids:
        values = pivot[formula_id].to_numpy(dtype=np.float64)
        difference = float(values.mean() - reference.mean())
        if formula_id == REFERENCE_FORMULA:
            p_value = adjusted_p = 1.0
            rank_biserial = 0.0
        else:
            p_value = tests[formula_id].p_value
            adjusted_p = adjusted[formula_id]
            rank_biserial = tests[formula_id].rank_biserial
        rows.append({
            "formula_id": formula_id,
            "technique": FORMULA_REGISTRY[formula_id].name,
            "mean_normalized_log_aulc": float(values.mean()),
            "harmonic_mean_normalized_log_aulc": float(reference.mean()),
            "difference_vs_harmonic_mean": difference,
            "wilcoxon_p_value": p_value,
            "holm_adjusted_p_value": adjusted_p,
            "rank_biserial": rank_biserial,
            **({"nonzero_pairs": 0 if formula_id == REFERENCE_FORMULA else
                tests[formula_id].nonzero_pairs} if include_counts else {}),
            "meaningful_difference": bool(
                formula_id != REFERENCE_FORMULA
                and adjusted_p < 0.05 and abs(difference) >= PRACTICAL_MARGIN),
        })
    return pd.DataFrame(rows), {
        "status": "complete",
        "reference": REFERENCE_FORMULA,
        "test": "paired Wilcoxon on normalized log-scale learning-curve area",
        "multiplicity": "Holm across 33 non-reference techniques",
        "practical_margin": PRACTICAL_MARGIN,
        "primary_blocks": 16,
    }


def conditional_coverage_analysis(metrics: pd.DataFrame, aggregate: Path) -> Tuple[pd.DataFrame, dict]:
    rows = []
    group_keys = ["voxel_size", "vla", "condition", "formula_id", "technique"]
    strata = ["configuration_id", "test_run", "training_size"]
    for identity, frame in metrics.groupby(group_keys, sort=False):
        try:
            result = fixed_effect_slope(
                frame,
                coverage_column="calibration_unique_coverage",
                outcome_column="roc_auc",
                strata_columns=strata,
            )
        except ValueError as error:
            if "no within-stratum variation" in str(error):
                continue
            raise
        row = dict(zip(group_keys, identity))
        row.update({
            "primary_block_id": str(row["vla"]) + "__" + str(row["condition"]),
            "roc_auc_per_coverage_unit": result.slope,
            "roc_auc_per_10pp_coverage": result.effect_per_10_percentage_points,
            "observations": result.observations,
            "informative_strata": result.informative_strata,
            "centered_coverage_sd": result.centered_coverage_sd,
        })
        rows.append(row)
    effects = pd.DataFrame(rows)
    effects.to_csv(aggregate / "within_size_coverage_effects.csv", index=False)
    if effects.empty:
        return effects, {
            "status": "not_run",
            "reason": "at least two repetitions with within-size coverage variation are required",
        }

    summary, metadata = conditional_coverage_tests(effects)
    if metadata["status"] == "complete":
        summary.to_csv(aggregate / "conditional_coverage_tests.csv", index=False)
    return effects, metadata


def conditional_coverage_tests(
    effects: pd.DataFrame, method: str = "approximate", include_counts: bool = False,
) -> Tuple[pd.DataFrame, dict]:
    """Test stored block slopes against zero; Holm spans the 34 techniques."""
    primary = effects[np.isclose(effects["voxel_size"].astype(float), 0.02)]
    formula_ids = list(STUDY_FORMULA_IDS)
    counts = primary.groupby("formula_id")["primary_block_id"].nunique()
    if any(int(counts.get(formula_id, 0)) != 16 for formula_id in formula_ids):
        return effects, {
            "status": "not_run",
            "reason": "complete 16-block conditional coverage slopes are required",
        }
    raw: Dict[str, float] = {}
    tests = {}
    for formula_id in formula_ids:
        values = primary.loc[
            primary["formula_id"] == formula_id, "roc_auc_per_10pp_coverage"
        ].to_numpy(dtype=np.float64)
        result = wilcoxon_signed_rank(values, np.zeros_like(values), method=method)
        raw[formula_id] = result.p_value
        tests[formula_id] = result
    adjusted = holm_adjust(raw)
    summary_rows = []
    for formula_id in formula_ids:
        selected = primary[primary["formula_id"] == formula_id]
        values = selected["roc_auc_per_10pp_coverage"].to_numpy(dtype=np.float64)
        summary_rows.append({
            "formula_id": formula_id,
            "technique": FORMULA_REGISTRY[formula_id].name,
            "mean_roc_auc_per_10pp_coverage": float(values.mean()),
            "median_roc_auc_per_10pp_coverage": float(np.median(values)),
            "positive_block_fraction": float(np.mean(values > 0)),
            "wilcoxon_p_value": tests[formula_id].p_value,
            "holm_adjusted_p_value": adjusted[formula_id],
            "rank_biserial": tests[formula_id].rank_biserial,
            "primary_blocks": len(values),
            **({"nonzero_pairs": tests[formula_id].nonzero_pairs} if include_counts else {}),
        })
    summary = pd.DataFrame(summary_rows)
    return summary, {
        "status": "complete",
        "test": "within-size fixed-effect coverage slopes tested against zero",
        "multiplicity": "Holm across 34 techniques",
        "primary_blocks": 16,
    }


def aggregate_results(output_root: Path) -> dict:
    aggregate = output_root / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)
    metric_paths = sorted((output_root / "cells").glob(
        "*/folds/*/*/rep_*/train_*/metrics.csv"))
    status_paths = sorted((output_root / "cells").glob(
        "*/folds/*/*/rep_*/train_*/cell.json"))
    if not metric_paths:
        raise ValueError("no completed RQ3 v2 metrics were found")
    metrics = pd.concat([pd.read_csv(path) for path in metric_paths], ignore_index=True)
    statuses = pd.DataFrame([
        json.loads(path.read_text(encoding="utf-8")) for path in status_paths])
    keys = [
        "voxel_size", "configuration_id", "test_run", "sampling_repetition",
        "training_size", "formula_id",
    ]
    if metrics.duplicated(keys).any():
        raise ValueError("aggregate input contains duplicate RQ3 v2 cells")
    critical = metrics[[
        "roc_auc", "average_precision", "calibration_unique_coverage",
        "calibration_mean_known_fraction", "test_unique_coverage",
        "test_mean_known_fraction",
    ]].to_numpy(dtype=np.float64)
    if not np.isfinite(critical).all():
        raise ValueError("aggregate input contains non-finite critical values")
    metrics.to_csv(aggregate / "cell_metrics.csv.gz", index=False, compression="gzip")
    statuses.sort_values([
        "voxel_size", "configuration_id", "test_run", "sampling_repetition",
        "training_size",
    ]).to_csv(aggregate / "cell_status.csv.gz", index=False, compression="gzip")

    run_path = output_root / "run.json"
    run_metadata = json.loads(run_path.read_text(encoding="utf-8")) if run_path.is_file() else {}
    signature_details = run_metadata.get("signature_details", {})
    training_sizes = tuple(signature_details.get("training_sizes", []))
    if len(training_sizes) < 2:
        raise ValueError("at least two training sizes are required for learning curves")

    block_curves, global_curves = aggregate_learning_curves(metrics, aggregate)
    unit_curves = learning_curve_units(metrics, training_sizes)
    unit_curves.to_csv(aggregate / "learning_curve_units.csv.gz", index=False, compression="gzip")
    block_efficiency, global_efficiency = aggregate_sample_efficiency(unit_curves, aggregate)

    adjacent, adjacent_metadata = adjacent_reference_tests(block_curves, training_sizes)
    adjacent.to_csv(aggregate / "adjacent_size_tests.csv", index=False)
    efficiency_tests, efficiency_metadata = sample_efficiency_tests(block_efficiency)
    efficiency_tests.to_csv(aggregate / "sample_efficiency_tests.csv", index=False)
    _, coverage_metadata = conditional_coverage_analysis(metrics, aggregate)
    statistical = {
        "adjacent_size_tests": adjacent_metadata,
        "sample_efficiency_tests": efficiency_metadata,
        "conditional_coverage_tests": coverage_metadata,
    }
    atomic_json(aggregate / "statistical_tests.json", statistical)

    expected_cells = int(run_metadata.get("planned_cells", 0))
    completed_cells = len(statuses)
    status = "complete" if expected_cells and completed_cells == expected_cells else "partial"
    primary_global = global_curves[
        np.isclose(global_curves["voxel_size"].astype(float), 0.02)]
    leaders = []
    for training_size, frame in primary_global.groupby("training_size", sort=True):
        leader = frame.sort_values(["roc_auc_rank", "formula_id"], kind="mergesort").iloc[0]
        leaders.append({
            "training_size": int(training_size),
            "formula_id": str(leader["formula_id"]),
            "technique": str(leader["technique"]),
            "macro_roc_auc": float(leader["roc_auc"]),
            "calibration_unique_coverage": float(leader["calibration_unique_coverage"]),
            "test_unique_coverage": float(leader["test_unique_coverage"]),
        })
    efficiency_primary = global_efficiency[
        np.isclose(global_efficiency["voxel_size"].astype(float), 0.02)]
    sample_leader = None
    if not efficiency_primary.empty:
        row = efficiency_primary.sort_values(
            ["naulc_rank", "formula_id"], kind="mergesort").iloc[0]
        sample_leader = {
            "formula_id": str(row["formula_id"]),
            "technique": str(row["technique"]),
            "normalized_log_aulc": float(row["normalized_log_aulc"]),
        }
    summary = {
        "status": status,
        "completed_at_utc": utc_now(),
        "phase": "trajectory-count learning curves with measured voxel coverage",
        "completed_cells": completed_cells,
        "expected_cells": expected_cells,
        "techniques": len(STUDY_FORMULA_IDS),
        "training_sizes": list(training_sizes),
        "leaders_by_training_size": leaders,
        "sample_efficiency_leader": sample_leader,
        "statistical_analysis": statistical,
        "interpretation_scope": (
            "trajectory-level failure discrimination; across-size changes combine "
            "additional observations and additional coverage"),
    }
    atomic_json(aggregate / "summary.json", summary)

    lines = [
        "# RQ3 v2 results",
        "",
        "Status: **{}** ({} of {} planned subset cells).".format(
            status, completed_cells, expected_cells or "unknown"),
        "",
        "| Spectrum trajectories | Leading technique | Macro ROC-AUC | "
        "Calibration coverage | Held-out coverage |",
        "|---:|---|---:|---:|---:|",
    ]
    for leader in leaders:
        lines.append("| {} | {} | {:.4f} | {:.1%} | {:.1%} |".format(
            leader["training_size"], leader["technique"], leader["macro_roc_auc"],
            leader["calibration_unique_coverage"], leader["test_unique_coverage"]))
    if sample_leader is not None:
        lines.extend([
            "",
            "The descriptive sample-efficiency leader was **{}** with normalized "
            "log-scale learning-curve area `{:.4f}`.".format(
                sample_leader["technique"], sample_leader["normalized_log_aulc"]),
        ])
    lines.extend([
        "",
        "Across-size improvements combine the effects of more spectrum observations "
        "and broader voxel coverage. The within-size analysis estimates the residual "
        "coverage association separately.",
    ])
    (aggregate / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def regenerate_statistical_review(aggregate: Path, method: str = "exact") -> dict:
    """Recompute paired tests across the frozen 4-VLA x 4-strategy design.

    Holm families of 4 adjacent changes, 33 efficiency contrasts and 34 stored
    coverage slopes are separate. Shared factors may limit block independence.
    Meaningful-difference labels are descriptive, not equivalence tests.
    """
    if method not in {"approximate", "exact"}:
        raise ValueError("method must be 'approximate' or 'exact'")
    aggregate = aggregate.expanduser().resolve()
    paths = {
        "learning": aggregate / "primary_block_learning_curves.csv",
        "efficiency": aggregate / "primary_block_sample_efficiency.csv",
        "coverage": aggregate / "within_size_coverage_effects.csv",
        "old_adjacent": aggregate / "adjacent_size_tests.csv",
        "old_efficiency": aggregate / "sample_efficiency_tests.csv",
        "old_coverage": aggregate / "conditional_coverage_tests.csv",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    learning, efficiency, coverage = [pd.read_csv(paths[key]) for key in
                                      ("learning", "efficiency", "coverage")]
    old_adjacent, old_efficiency, old_coverage = [pd.read_csv(paths[key]) for key in
                                                  ("old_adjacent", "old_efficiency", "old_coverage")]
    for frame, outcome in ((learning, "roc_auc"), (efficiency, "normalized_log_aulc"),
                           (coverage, "roc_auc_per_10pp_coverage")):
        if set(frame.voxel_size) != {0.02}:
            raise ValueError("the frozen RQ4 review requires voxel size 0.02")
        if frame is learning:
            if set(frame.training_size) != set(DEFAULT_TRAINING_SIZES):
                raise ValueError("unexpected training sizes")
            for _, group in frame.groupby("training_size"):
                validate_primary_block_metrics(group)
        else:
            validate_primary_block_metrics(frame.rename(columns={outcome: "roc_auc"}))
    adjacent, adjacent_meta = adjacent_reference_tests(
        learning, DEFAULT_TRAINING_SIZES, method=method, include_counts=True)
    sample, sample_meta = sample_efficiency_tests(
        efficiency, method=method, include_counts=True)
    conditional, conditional_meta = conditional_coverage_tests(
        coverage.sort_values(["primary_block_id", "formula_id"]),
        method=method, include_counts=True)
    if any(meta["status"] != "complete" for meta in
           (adjacent_meta, sample_meta, conditional_meta)):
        raise ValueError("incomplete review families")
    outputs = {
        f"adjacent_size_tests_{method}.csv": adjacent,
        f"sample_efficiency_tests_{method}.csv": sample.sort_values("formula_id"),
        f"conditional_coverage_tests_{method}.csv": conditional.sort_values("formula_id"),
    }
    for frame, family in zip(outputs.values(), (4, 33, 34)):
        frame["wilcoxon_method"] = method
        frame["holm_family_size"] = family
    # Reconcile descriptive quantities before writing any derived output.
    for frame, historical, keys in zip(outputs.values(),
            (old_adjacent, old_efficiency, old_coverage),
            (["lower_training_size", "upper_training_size"], ["formula_id"], ["formula_id"])):
        left = frame.set_index(keys).sort_index()
        right = historical.set_index(keys).sort_index()
        if not left.index.equals(right.index):
            raise ValueError("historical comparison identities do not match aggregates")
        if set(right.columns) - set(left.columns):
            raise ValueError("review is missing required historical columns")
        for column in right.select_dtypes(include=[np.number]).columns:
            if column in {"wilcoxon_p_value", "holm_adjusted_p_value"}:
                continue
            if not np.allclose(left[column], right[column], rtol=0, atol=1e-14):
                raise ValueError("historical values do not reconcile: " + column)
        frame["significant_0p05"] = frame["holm_adjusted_p_value"] < 0.05
    for name, frame in outputs.items():
        frame.to_csv(aggregate / name, index=False)
    summary={"status":"complete","wilcoxon_method":method,"primary_blocks":16,
        "families":{"adjacent":4,"sample_efficiency":33,"conditional_coverage":34},
        "nonzero_pair_max": max(int(frame.nonzero_pairs.max()) for frame in outputs.values()),
        "outputs":list(outputs),"sources_sha256":{p.name:sha256_file(p) for p in paths.values()},
        "output_sha256": {name: sha256_file(aggregate/name) for name in outputs},
        "wilcoxon_alternative": "two-sided",
        "z_score_reported":False}
    atomic_json(aggregate/f"statistical_review_{method}.json",summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("input"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--voxel-size", type=float, action="append", dest="voxel_sizes")
    parser.add_argument("--origin", type=float, nargs=3, default=(-1.0, -1.0, 0.0))
    parser.add_argument("--training-size", type=int, action="append", dest="training_sizes")
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--sampling-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--chunk-size", type=int, default=250_000)
    parser.add_argument("--configuration", action="append", default=[])
    parser.add_argument("--max-folds", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument("--statistical-review-only", action="store_true")
    parser.add_argument("--wilcoxon-method", choices=("approximate", "exact"), default="exact")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    voxel_sizes = tuple(args.voxel_sizes or DEFAULT_VOXEL_SIZES)
    training_sizes = tuple(sorted(args.training_sizes or DEFAULT_TRAINING_SIZES))
    if any(value <= 0 for value in voxel_sizes):
        raise ValueError("voxel sizes must be positive")
    if len(set(voxel_sizes)) != len(voxel_sizes):
        raise ValueError("voxel sizes must not contain duplicates")
    if any(value < 2 or value % 2 for value in training_sizes):
        raise ValueError("training sizes must be even integers of at least two")
    if len(set(training_sizes)) != len(training_sizes) or len(training_sizes) < 2:
        raise ValueError("at least two distinct training sizes are required")
    if args.repetitions < 1:
        raise ValueError("repetitions must be positive")
    if args.max_folds is not None and not 1 <= args.max_folds <= len(EXPECTED_RUNS):
        raise ValueError("max-folds must be between 1 and 10")

    if args.output_root is None:
        relative = Path("experiments/RQ3_v2/results/smoke") if args.smoke else Path(
            "experiments/RQ3_v2/results/learning_curves")
        output_root = (REPOSITORY_ROOT / relative).resolve()
    else:
        output_root = args.output_root.expanduser().resolve()
    if args.statistical_review_only:
        if args.analysis_only or args.smoke:
            raise ValueError("statistical review mode is exclusive with experiment modes")
        result = regenerate_statistical_review(
            output_root / "aggregate", args.wilcoxon_method)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.analysis_only:
        print(json.dumps(aggregate_results(output_root), indent=2, sort_keys=True))
        return 0

    configurations = list(discover_configurations(args.input_root))
    if args.configuration:
        wanted = set(args.configuration)
        configurations = [item for item in configurations if item.configuration_id in wanted]
        missing = wanted - {item.configuration_id for item in configurations}
        if missing:
            raise ValueError("unknown configurations: " + ", ".join(sorted(missing)))
    if args.smoke and not args.configuration:
        configurations = [
            item for item in configurations if item.configuration_id == "EO1__DH__200"]
    if not configurations:
        raise ValueError("no configurations selected")
    folds_per_configuration = args.max_folds or len(EXPECTED_RUNS)
    repetitions = 1 if args.smoke and args.repetitions == DEFAULT_REPETITIONS else args.repetitions
    if args.smoke and args.max_folds is None:
        folds_per_configuration = 1

    signature, signature_details = experiment_signature(
        voxel_sizes=voxel_sizes,
        origin=args.origin,
        training_sizes=training_sizes,
        repetitions=repetitions,
        sampling_seed=args.sampling_seed,
        configuration_ids=[item.configuration_id for item in configurations],
        folds_per_configuration=folds_per_configuration,
    )
    planned = (
        len(configurations) * len(voxel_sizes) * folds_per_configuration
        * repetitions * len(training_sizes))
    output_root.mkdir(parents=True, exist_ok=True)
    run_path = output_root / "run.json"
    if run_path.is_file():
        existing = json.loads(run_path.read_text(encoding="utf-8"))
        if existing.get("experiment_signature") != signature:
            raise ValueError("output root belongs to a different RQ3 v2 design")
        if not args.resume:
            raise FileExistsError("RQ3 v2 output exists and resume is disabled")
    run_metadata = {
        "status": "running",
        "started_at_utc": utc_now(),
        "experiment": "RQ3_v2_learning_curves",
        "experiment_signature": signature,
        "signature_details": signature_details,
        "planned_cells": planned,
        "input_root": str(args.input_root.expanduser().resolve()),
        "output_root": str(output_root),
        "python": sys.version,
        "platform": platform.platform(),
        "git_revision": git_revision(),
    }
    atomic_json(run_path, run_metadata)
    atomic_json(output_root / "audit" / "design.json", signature_details)

    started = time.perf_counter()
    completed = 0
    for voxel_size in voxel_sizes:
        grid = VoxelGrid(voxel_size=voxel_size, origin=tuple(args.origin))
        for configuration in configurations:
            print("[load] {} size={:.2f}".format(
                configuration.configuration_id, voxel_size), flush=True)
            dataset = load_trajectory_dataset(
                [configuration.directory], grid=grid, require_labels=True,
                chunk_size=args.chunk_size)
            runs = list(validate_configuration_metadata(configuration, dataset.episodes))
            selected_test_runs = runs[:folds_per_configuration]
            for test_run in selected_test_runs:
                calibration_run = choose_calibration_run(test_run, runs)
                run_values = dataset.episodes["run_id"].astype(str)
                testing = subset_dataset(
                    dataset, dataset.episodes.loc[run_values == test_run, "episode_id"])
                calibration = subset_dataset(
                    dataset, dataset.episodes.loc[run_values == calibration_run, "episode_id"])
                pool = subset_dataset(
                    dataset, dataset.episodes.loc[
                        ~run_values.isin([test_run, calibration_run]), "episode_id"])
                if (
                    set(testing.episodes["episode_id"].astype(str))
                    & set(calibration.episodes["episode_id"].astype(str))
                ):
                    raise ValueError("calibration and evaluation runs overlap")
                for repetition in range(1, repetitions + 1):
                    seed = stable_seed(
                        args.sampling_seed, voxel_size,
                        configuration.configuration_id, test_run, repetition)
                    suites = nested_balanced_subsets(
                        pool.episodes, training_sizes, np.random.default_rng(seed))
                    audit_directory = (
                        output_root / "cells" / size_slug(voxel_size) / "folds"
                        / configuration.configuration_id / test_run
                        / "rep_{:02d}".format(repetition))
                    write_nested_suite_audit(audit_directory, pool, suites)
                    for training_size in training_sizes:
                        completed += 1
                        result = run_learning_curve_cell(
                            pool=pool,
                            calibration=calibration,
                            testing=testing,
                            configuration=configuration,
                            test_run=test_run,
                            calibration_run=calibration_run,
                            grid=grid,
                            output_root=output_root,
                            signature=signature,
                            training_size=training_size,
                            episode_ids=suites[training_size],
                            repetition=repetition,
                            resume=args.resume,
                            save_models=args.save_models,
                            save_predictions=args.save_predictions,
                        )
                        print(
                            "[cell {}/{}] {} {} rep={} n={} calibration={:.3f} "
                            "test={:.3f}".format(
                                completed, planned, configuration.configuration_id,
                                test_run, repetition, training_size,
                                result["calibration_unique_coverage"],
                                result["test_unique_coverage"]),
                            flush=True,
                        )

    print("[aggregate] computing learning curves and sample efficiency", flush=True)
    summary = aggregate_results(output_root)
    run_metadata.update({
        "status": summary["status"],
        "completed_at_utc": utc_now(),
        "elapsed_seconds": time.perf_counter() - started,
        "summary": summary,
    })
    atomic_json(run_path, run_metadata)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
