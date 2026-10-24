#!/usr/bin/env python3
"""Run the RQ3 coverage-controlled trajectory-subset experiment."""

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

from experiments.RQ1.analysis import holm_adjust, wilcoxon_signed_rank
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
)
from experiments.RQ3.analysis import (
    coverage_statistics,
    matched_high_low_effects,
    select_coverage_quantiles,
    voxel_union,
)
from vla_sbfl.dataset import TrajectoryDataset, load_trajectory_dataset
from vla_sbfl.formulas import FORMULA_REGISTRY, TABLE4_FORMULA_IDS
from vla_sbfl.metrics import evaluate_predictions
from vla_sbfl.model import SBFLModel, episode_id_hash
from vla_sbfl.scoring import TrajectoryScorer
from vla_sbfl.voxelizer import VoxelGrid


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VOXEL_SIZES = (0.02,)
DEFAULT_TRAINING_SIZES = (10, 20, 50, 100, 200)
DEFAULT_COVERAGE_TARGETS = {"low": 0.10, "medium": 0.50, "high": 0.90}
DEFAULT_CANDIDATE_SUBSETS = 200
DEFAULT_SELECTION_REPLICATES = 1
REFERENCE_FORMULA = "harmonic_mean"
PRACTICAL_MARGIN = 0.01


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def size_slug(voxel_size: float) -> str:
    return "size_" + format(float(voxel_size), ".2f").replace(".", "p")


def training_slug(training_size: int) -> str:
    return "train_{:04d}".format(int(training_size))


def sha256_payload(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = "\x1f".join([str(int(base_seed)), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def subset_sha256(episode_ids: Iterable[str]) -> str:
    return episode_id_hash(tuple(sorted(str(value) for value in episode_ids)))


def balanced_sample_ids(
    episodes: pd.DataFrame,
    sample_size: int,
    generator: np.random.Generator,
) -> Tuple[str, ...]:
    """Draw an exactly class-balanced subset from a training pool."""

    if sample_size < 2 or sample_size % 2:
        raise ValueError("training sizes must be even and at least two")
    if "failure" not in episodes.columns or "episode_id" not in episodes.columns:
        raise ValueError("episodes must contain episode_id and failure")
    per_class = sample_size // 2
    selected: List[str] = []
    for label in (0, 1):
        available = np.asarray(sorted(
            episodes.loc[episodes["failure"].astype(int) == label, "episode_id"].astype(str)
        ), dtype=object)
        if len(available) < per_class:
            raise ValueError(
                "training pool lacks {} trajectories for class {}".format(per_class, label))
        selected.extend(str(value) for value in generator.choice(
            available, size=per_class, replace=False))
    return tuple(sorted(selected))


def choose_calibration_run(test_run: str, runs: Sequence[str]) -> str:
    """Rotate a non-test run into calibration without inspecting outcomes."""

    ordered = tuple(sorted((str(value) for value in runs), key=run_sort_key))
    if test_run not in ordered or len(ordered) < 3:
        raise ValueError("test run must belong to a sequence of at least three runs")
    index = ordered.index(test_run)
    calibration = ordered[(index + 1) % len(ordered)]
    if calibration == test_run:
        raise AssertionError("calibration run cannot equal the test run")
    return calibration


def candidate_coverage_table(
    pool: TrajectoryDataset,
    calibration: TrajectoryDataset,
    training_size: int,
    candidate_subsets: int,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, Tuple[str, ...]]]:
    """Generate balanced random candidates and measure calibration support."""

    if candidate_subsets < len(DEFAULT_COVERAGE_TARGETS):
        raise ValueError("candidate_subsets must permit three distinct targets")
    generator = np.random.default_rng(seed)
    calibration_support = voxel_union(calibration.visits)
    pool_support = voxel_union(pool.visits)
    if not calibration_support or not pool_support:
        raise ValueError("pool and calibration support must not be empty")

    rows = []
    subsets: Dict[str, Tuple[str, ...]] = {}
    attempts = 0
    maximum_attempts = candidate_subsets * 20
    while len(rows) < candidate_subsets and attempts < maximum_attempts:
        attempts += 1
        episode_ids = balanced_sample_ids(pool.episodes, training_size, generator)
        identity = subset_sha256(episode_ids)
        if identity in subsets:
            continue
        support = voxel_union({episode_id: pool.visits[episode_id] for episode_id in episode_ids})
        selected_episodes = pool.episodes[
            pool.episodes["episode_id"].astype(str).isin(episode_ids)]
        numeric_steps = pd.to_numeric(
            selected_episodes.get("num_steps", pd.Series(dtype=float)), errors="coerce")
        rows.append({
            "candidate_index": len(rows),
            "subset_sha256": identity,
            "training_size": int(training_size),
            "training_failures": int(selected_episodes["failure"].sum()),
            "training_passes": int((selected_episodes["failure"] == 0).sum()),
            "training_voxels": len(support),
            "calibration_unique_coverage": (
                len(support & calibration_support) / len(calibration_support)),
            "training_pool_unique_coverage": len(support) / len(pool_support),
            "object_count": (
                int(selected_episodes["object_name"].astype(str).nunique())
                if "object_name" in selected_episodes.columns else 0),
            "mean_num_steps": (
                float(numeric_steps.mean()) if len(numeric_steps) else float("nan")),
        })
        subsets[identity] = episode_ids
    if len(rows) != candidate_subsets:
        raise ValueError("could not generate enough unique candidate subsets")
    return pd.DataFrame(rows), subsets


def experiment_signature(
    voxel_sizes: Sequence[float],
    origin: Sequence[float],
    training_sizes: Sequence[int],
    coverage_targets: Mapping[str, float],
    candidate_subsets: int,
    selection_replicates: int,
    selection_seed: int,
) -> Tuple[str, dict]:
    directory = Path(__file__).resolve().parent
    catalogue = [{
        "formula_id": formula_id,
        "name": FORMULA_REGISTRY[formula_id].name,
        "expression": FORMULA_REGISTRY[formula_id].expression,
    } for formula_id in TABLE4_FORMULA_IDS]
    details = {
        "protocol_sha256": sha256_file(directory / "RQ3.md"),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "analysis_sha256": sha256_file(directory / "analysis.py"),
        "voxel_sizes": [float(value) for value in voxel_sizes],
        "origin": [float(value) for value in origin],
        "coordinate_frame": "robot_base",
        "training_sizes": [int(value) for value in training_sizes],
        "coverage_targets": dict(coverage_targets),
        "coverage_selection_metric": "calibration_unique_coverage",
        "candidate_subsets": int(candidate_subsets),
        "selection_replicates": int(selection_replicates),
        "selection_seed": int(selection_seed),
        "class_balance": "exactly 50% pass and 50% fail",
        "split": "one test run, next cyclic run calibration, remaining eight runs pool",
        "formula_catalogue": catalogue,
        "trajectory_aggregation": "maximum_training_voxel_empirical_percentile",
        "all_unknown_policy": "neutral_0.5",
    }
    return sha256_payload(details), details


def cell_directory(
    output_root: Path,
    voxel_size: float,
    configuration: Configuration,
    test_run: str,
    training_size: int,
    replicate: int,
    coverage_tier: str,
) -> Path:
    return (
        output_root / "cells" / size_slug(voxel_size) / "folds"
        / configuration.configuration_id / test_run / training_slug(training_size)
        / "rep_{:02d}".format(replicate) / coverage_tier
    )


def run_selected_cell(
    pool: TrajectoryDataset,
    calibration: TrajectoryDataset,
    testing: TrajectoryDataset,
    configuration: Configuration,
    test_run: str,
    calibration_run: str,
    grid: VoxelGrid,
    output_root: Path,
    signature: str,
    selected: Mapping[str, object],
    episode_ids: Sequence[str],
    replicate: int,
    resume: bool,
    save_models: bool,
    save_predictions: bool,
) -> dict:
    training_size = int(selected["training_size"])
    coverage_tier = str(selected["coverage_tier"])
    destination = cell_directory(
        output_root, grid.voxel_size, configuration, test_run,
        training_size, replicate, coverage_tier)
    completion_path = destination / "cell.json"
    if completion_path.is_file():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("experiment_signature") != signature:
            raise ValueError("completed cell has a different signature: " + str(destination))
        if not resume:
            raise FileExistsError("completed cell exists and resume is disabled: " + str(destination))
        return completion

    training = subset_dataset(pool, episode_ids)
    if set(training.labels.values()) != {0, 1} or len(training.episodes) != training_size:
        raise ValueError("selected training subset is incomplete or lacks both classes")
    if set(training.episodes["episode_id"].astype(str)) & set(testing.episodes["episode_id"].astype(str)):
        raise ValueError("training and test trajectories overlap")
    if set(training.episodes["episode_id"].astype(str)) & set(calibration.episodes["episode_id"].astype(str)):
        raise ValueError("training and calibration trajectories overlap")

    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"episode_id": sorted(episode_ids)}).to_csv(
        destination / "training_episode_ids.csv.gz", index=False, compression="gzip")
    support = voxel_union(training.visits)
    calibration_coverage = coverage_statistics(support, calibration.visits, calibration.labels)
    test_coverage = coverage_statistics(support, testing.visits, testing.labels)

    started = time.perf_counter()
    fit_started = time.perf_counter()
    model = SBFLModel.fit(
        visits=training.visits,
        labels=training.labels,
        grid=grid,
        formula_ids=TABLE4_FORMULA_IDS,
        metadata={
            "experiment": "RQ3_phase_a_coverage",
            "experiment_signature": signature,
            "configuration_id": configuration.configuration_id,
            "vla": configuration.vla,
            "condition": configuration.condition,
            "budget": configuration.budget,
            "test_run": test_run,
            "calibration_run": calibration_run,
            "training_size": training_size,
            "coverage_tier": coverage_tier,
            "target_quantile": float(selected["target_quantile"]),
            "selection_replicate": replicate,
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
        "training_size": training_size,
        "coverage_tier": coverage_tier,
        "target_quantile": float(selected["target_quantile"]),
        "selection_replicate": int(replicate),
        "subset_sha256": str(selected["subset_sha256"]),
    }
    for position, (column, value) in enumerate(reversed(list(identifiers.items()))):
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


def aggregate_results(output_root: Path) -> dict:
    aggregate = output_root / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)
    metric_paths = sorted((output_root / "cells").glob(
        "*/folds/*/*/train_*/rep_*/*/metrics.csv"))
    status_paths = sorted((output_root / "cells").glob(
        "*/folds/*/*/train_*/rep_*/*/cell.json"))
    if not metric_paths:
        raise ValueError("no completed RQ3 metrics were found")
    metrics = pd.concat([pd.read_csv(path) for path in metric_paths], ignore_index=True)
    statuses = pd.DataFrame([
        json.loads(path.read_text(encoding="utf-8")) for path in status_paths])
    keys = [
        "voxel_size", "configuration_id", "test_run", "training_size",
        "selection_replicate", "coverage_tier", "formula_id",
    ]
    if metrics.duplicated(keys).any():
        raise ValueError("aggregate input contains duplicate RQ3 cells")
    numeric = metrics[[
        "roc_auc", "average_precision", "calibration_unique_coverage",
        "calibration_mean_known_fraction", "test_unique_coverage",
        "test_mean_known_fraction",
    ]].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("RQ3 aggregate input contains non-finite critical values")
    metrics.to_csv(aggregate / "cell_metrics.csv.gz", index=False, compression="gzip")
    statuses.sort_values([
        "voxel_size", "configuration_id", "test_run", "training_size",
        "selection_replicate", "coverage_tier",
    ]).to_csv(aggregate / "cell_status.csv.gz", index=False, compression="gzip")

    measure_columns = {
        "roc_auc": ("roc_auc", "mean"),
        "average_precision": ("average_precision", "mean"),
        "calibration_unique_coverage": ("calibration_unique_coverage", "mean"),
        "calibration_mean_known_fraction": ("calibration_mean_known_fraction", "mean"),
        "test_unique_coverage": ("test_unique_coverage", "mean"),
        "test_mean_known_fraction": ("test_mean_known_fraction", "mean"),
        "test_failure_known_fraction": ("test_failure_known_fraction", "mean"),
        "test_pass_known_fraction": ("test_pass_known_fraction", "mean"),
        "training_voxels": ("training_voxels", "mean"),
        "all_unknown_trajectories": ("all_unknown_trajectories", "sum"),
        "score_tie_fraction": ("score_tie_fraction", "mean"),
    }
    configuration = metrics.groupby([
        "voxel_size", "configuration_id", "vla", "condition", "budget",
        "training_size", "coverage_tier", "formula_id", "technique",
    ], as_index=False).agg(**measure_columns)
    configuration.to_csv(aggregate / "configuration_metrics.csv", index=False)

    block = configuration.groupby([
        "voxel_size", "vla", "condition", "training_size", "coverage_tier",
        "formula_id", "technique",
    ], as_index=False).agg(**measure_columns)
    block["primary_block_id"] = block["vla"].astype(str) + "__" + block["condition"].astype(str)
    block.to_csv(aggregate / "primary_block_metrics.csv", index=False)

    by_size = block.groupby([
        "voxel_size", "training_size", "coverage_tier", "formula_id", "technique",
    ], as_index=False).agg(
        macro_roc_auc=("roc_auc", "mean"),
        macro_average_precision=("average_precision", "mean"),
        mean_calibration_unique_coverage=("calibration_unique_coverage", "mean"),
        mean_test_unique_coverage=("test_unique_coverage", "mean"),
        mean_test_known_fraction=("test_mean_known_fraction", "mean"),
        primary_blocks=("primary_block_id", "nunique"),
    )
    by_size["roc_auc_rank"] = by_size.groupby([
        "voxel_size", "training_size", "coverage_tier"
    ])["macro_roc_auc"].rank(method="average", ascending=False)
    by_size.to_csv(aggregate / "coverage_tier_metrics.csv", index=False)

    rankings = block.groupby([
        "voxel_size", "coverage_tier", "formula_id", "technique",
    ], as_index=False).agg(
        macro_roc_auc=("roc_auc", "mean"),
        macro_average_precision=("average_precision", "mean"),
        mean_test_unique_coverage=("test_unique_coverage", "mean"),
        mean_test_known_fraction=("test_mean_known_fraction", "mean"),
        primary_blocks=("primary_block_id", "nunique"),
        training_sizes=("training_size", "nunique"),
    )
    rankings["rank"] = rankings.groupby([
        "voxel_size", "coverage_tier"
    ])["macro_roc_auc"].rank(method="average", ascending=False)
    rankings = rankings.sort_values([
        "voxel_size", "coverage_tier", "rank", "formula_id"])
    rankings.to_csv(aggregate / "rankings_by_coverage.csv", index=False)

    effects = matched_high_low_effects(metrics)
    effects.to_csv(aggregate / "matched_high_low_cells.csv.gz", index=False, compression="gzip")
    configuration_effects = effects.groupby([
        "voxel_size", "configuration_id", "vla", "condition", "budget",
        "training_size", "formula_id", "technique",
    ], as_index=False).agg(
        delta_roc_auc=("delta_roc_auc", "mean"),
        delta_average_precision=("delta_average_precision", "mean"),
        delta_calibration_coverage=("delta_calibration_mean_known_fraction", "mean"),
        delta_test_coverage=("delta_test_mean_known_fraction", "mean"),
        delta_test_unique_coverage=("delta_test_unique_coverage", "mean"),
    )
    block_effects = configuration_effects.groupby([
        "voxel_size", "vla", "condition", "training_size", "formula_id", "technique",
    ], as_index=False).agg(
        delta_roc_auc=("delta_roc_auc", "mean"),
        delta_average_precision=("delta_average_precision", "mean"),
        delta_calibration_coverage=("delta_calibration_coverage", "mean"),
        delta_test_coverage=("delta_test_coverage", "mean"),
        delta_test_unique_coverage=("delta_test_unique_coverage", "mean"),
    )
    block_effects["primary_block_id"] = (
        block_effects["vla"].astype(str) + "__" + block_effects["condition"].astype(str))
    block_effects.to_csv(aggregate / "matched_coverage_effects.csv", index=False)

    interactions_rows = []
    test_payload: dict = {
        "status": "not_run",
        "reason": "the complete matched primary design is required",
    }
    primary = block_effects[np.isclose(block_effects["voxel_size"], 0.02)].groupby([
        "primary_block_id", "formula_id", "technique",
    ], as_index=False).agg(delta_roc_auc=("delta_roc_auc", "mean"))
    pivot = primary.pivot(index="primary_block_id", columns="formula_id", values="delta_roc_auc")
    formula_ids = list(TABLE4_FORMULA_IDS)
    if (
        pivot.shape[0] == 16
        and all(formula_id in pivot.columns for formula_id in formula_ids)
        and not pivot[formula_ids].isna().any().any()
    ):
        reference = pivot[REFERENCE_FORMULA].to_numpy(dtype=np.float64)
        raw_p_values = {}
        tests = {}
        for formula_id in formula_ids:
            if formula_id == REFERENCE_FORMULA:
                continue
            result = wilcoxon_signed_rank(
                pivot[formula_id].to_numpy(dtype=np.float64), reference)
            raw_p_values[formula_id] = result.p_value
            tests[formula_id] = result
        adjusted = holm_adjust(raw_p_values)
        reference_mean = float(reference.mean())
        for formula_id in formula_ids:
            values = pivot[formula_id].to_numpy(dtype=np.float64)
            interaction = float(values.mean() - reference_mean)
            if formula_id == REFERENCE_FORMULA:
                raw_p = adjusted_p = 1.0
                rank_biserial = 0.0
            else:
                raw_p = tests[formula_id].p_value
                adjusted_p = adjusted[formula_id]
                rank_biserial = tests[formula_id].rank_biserial
            interactions_rows.append({
                "formula_id": formula_id,
                "technique": FORMULA_REGISTRY[formula_id].name,
                "mean_high_minus_low_auc": float(values.mean()),
                "reference_high_minus_low_auc": reference_mean,
                "interaction_vs_harmonic_mean": interaction,
                "wilcoxon_p_value": raw_p,
                "holm_adjusted_p_value": adjusted_p,
                "rank_biserial": rank_biserial,
                "meaningful_interaction": bool(
                    formula_id != REFERENCE_FORMULA
                    and adjusted_p < 0.05
                    and abs(interaction) >= PRACTICAL_MARGIN),
            })
        test_payload = {
            "status": "complete",
            "test": "paired Wilcoxon on high-minus-low effects versus Harmonic Mean",
            "multiplicity": "Holm correction across 30 non-reference techniques",
            "primary_blocks": 16,
            "coverage_effect_practical_margin": PRACTICAL_MARGIN,
        }
    interactions = pd.DataFrame(interactions_rows)
    interactions.to_csv(aggregate / "technique_coverage_interactions.csv", index=False)
    atomic_json(aggregate / "statistical_tests.json", test_payload)

    run_path = output_root / "run.json"
    run_metadata = json.loads(run_path.read_text(encoding="utf-8")) if run_path.is_file() else {}
    signature_details = run_metadata.get("signature_details", {})
    expected_cells = (
        32 * len(EXPECTED_RUNS)
        * len(signature_details.get("voxel_sizes", []))
        * len(signature_details.get("training_sizes", []))
        * int(signature_details.get("selection_replicates", 0))
        * len(signature_details.get("coverage_targets", {}))
    )
    completed_cells = int(len(statuses))
    status = "complete" if expected_cells and completed_cells == expected_cells else "partial"
    leaders = []
    if not rankings.empty:
        for (voxel_size, tier), frame in rankings.groupby(["voxel_size", "coverage_tier"]):
            leader = frame.sort_values(["rank", "formula_id"]).iloc[0]
            leaders.append({
                "voxel_size": float(voxel_size),
                "coverage_tier": str(tier),
                "formula_id": str(leader["formula_id"]),
                "technique": str(leader["technique"]),
                "macro_roc_auc": float(leader["macro_roc_auc"]),
            })
    summary = {
        "status": status,
        "completed_at_utc": utc_now(),
        "phase": "A: held-out trajectory discrimination under controlled voxel coverage",
        "completed_cells": completed_cells,
        "expected_cells": expected_cells,
        "techniques": len(TABLE4_FORMULA_IDS),
        "leaders": leaders,
        "statistical_analysis": test_payload,
        "selection_status": (
            "trajectory-level failure discrimination; causal voxel localization remains Phase B"),
    }
    atomic_json(aggregate / "summary.json", summary)
    lines = [
        "# RQ3 Phase A results",
        "",
        "Status: **{}** ({} of {} planned cells).".format(
            status, completed_cells, expected_cells or "unknown"),
        "",
        "Training subsets are outcome-balanced and selected by calibration coverage; "
        "the final test run is not used for subset selection.",
        "",
        "| Voxel size | Coverage tier | Leading technique | Macro ROC-AUC |",
        "|---:|---|---|---:|",
    ]
    for leader in leaders:
        lines.append("| {:.2f} | {} | {} | {:.4f} |".format(
            leader["voxel_size"], leader["coverage_tier"],
            leader["technique"], leader["macro_roc_auc"]))
    lines.extend([
        "",
        "These are trajectory-level discrimination results. They do not identify "
        "causal fault voxels without independent spatial ground truth.",
    ])
    (aggregate / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT,
            stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("input"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--voxel-size", type=float, action="append", dest="voxel_sizes")
    parser.add_argument("--origin", type=float, nargs=3, default=(-1.0, -1.0, 0.0))
    parser.add_argument("--training-size", type=int, action="append", dest="training_sizes")
    parser.add_argument(
        "--coverage-quantiles", type=float, nargs=3, metavar=("LOW", "MEDIUM", "HIGH"),
        default=(0.10, 0.50, 0.90))
    parser.add_argument("--candidate-subsets", type=int, default=DEFAULT_CANDIDATE_SUBSETS)
    parser.add_argument("--selection-replicates", type=int, default=DEFAULT_SELECTION_REPLICATES)
    parser.add_argument("--selection-seed", type=int, default=20261023)
    parser.add_argument("--chunk-size", type=int, default=250_000)
    parser.add_argument("--configuration", action="append", default=[])
    parser.add_argument("--max-folds", type=int)
    parser.add_argument("--max-training-sizes", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    voxel_sizes = tuple(args.voxel_sizes or DEFAULT_VOXEL_SIZES)
    training_sizes = tuple(args.training_sizes or DEFAULT_TRAINING_SIZES)
    if any(value <= 0 for value in voxel_sizes):
        raise ValueError("voxel sizes must be positive")
    if any(value < 2 or value % 2 for value in training_sizes):
        raise ValueError("training sizes must be positive even integers")
    if len(set(voxel_sizes)) != len(voxel_sizes) or len(set(training_sizes)) != len(training_sizes):
        raise ValueError("voxel and training sizes must not contain duplicates")
    quantile_values = tuple(float(value) for value in args.coverage_quantiles)
    if not (0.0 <= quantile_values[0] < quantile_values[1] < quantile_values[2] <= 1.0):
        raise ValueError("coverage quantiles must be strictly increasing in [0, 1]")
    coverage_targets = dict(zip(("low", "medium", "high"), quantile_values))
    if args.candidate_subsets < 3 or args.selection_replicates < 1:
        raise ValueError("candidate subsets must be >= 3 and replicates must be >= 1")
    if args.max_folds is not None and not 1 <= args.max_folds <= len(EXPECTED_RUNS):
        raise ValueError("max-folds must be between 1 and 10")
    if args.max_training_sizes is not None:
        if args.max_training_sizes < 1:
            raise ValueError("max-training-sizes must be positive")
        training_sizes = training_sizes[:args.max_training_sizes]
    if args.smoke:
        if args.max_folds is None:
            args.max_folds = 1
        if args.max_training_sizes is None:
            training_sizes = training_sizes[:1]
        if args.candidate_subsets == DEFAULT_CANDIDATE_SUBSETS:
            args.candidate_subsets = 20

    if args.output_root is None:
        relative = Path("experiments/RQ3/results/smoke") if args.smoke else Path(
            "experiments/RQ3/results/phase_a_coverage")
        output_root = (REPOSITORY_ROOT / relative).resolve()
    else:
        output_root = args.output_root.expanduser().resolve()
    if args.analysis_only:
        print(json.dumps(aggregate_results(output_root), indent=2, sort_keys=True))
        return 0

    signature, signature_details = experiment_signature(
        voxel_sizes=voxel_sizes,
        origin=args.origin,
        training_sizes=training_sizes,
        coverage_targets=coverage_targets,
        candidate_subsets=args.candidate_subsets,
        selection_replicates=args.selection_replicates,
        selection_seed=args.selection_seed,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    existing_run = output_root / "run.json"
    if existing_run.is_file():
        existing = json.loads(existing_run.read_text(encoding="utf-8"))
        if existing.get("experiment_signature") != signature:
            raise ValueError("output root belongs to a different RQ3 design")
        if not args.resume:
            raise FileExistsError("RQ3 output exists and resume is disabled")
    run_metadata = {
        "status": "running",
        "started_at_utc": utc_now(),
        "experiment": "RQ3_phase_a_coverage",
        "experiment_signature": signature,
        "signature_details": signature_details,
        "input_root": str(args.input_root.expanduser().resolve()),
        "output_root": str(output_root),
        "python": sys.version,
        "platform": platform.platform(),
        "git_revision": git_revision(),
    }
    atomic_json(existing_run, run_metadata)
    atomic_json(output_root / "audit" / "design.json", signature_details)

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

    started = time.perf_counter()
    completed = 0
    planned = (
        len(configurations) * len(voxel_sizes)
        * (args.max_folds or len(EXPECTED_RUNS)) * len(training_sizes)
        * args.selection_replicates * len(coverage_targets))
    for voxel_size in voxel_sizes:
        grid = VoxelGrid(voxel_size=voxel_size, origin=tuple(args.origin))
        for configuration in configurations:
            print("[load] {} size={:.2f}".format(
                configuration.configuration_id, voxel_size), flush=True)
            dataset = load_trajectory_dataset(
                [configuration.directory], grid=grid, require_labels=True,
                chunk_size=args.chunk_size)
            runs = list(validate_configuration_metadata(configuration, dataset.episodes))
            selected_test_runs = runs[:args.max_folds] if args.max_folds else runs
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
                for training_size in training_sizes:
                    for replicate in range(1, args.selection_replicates + 1):
                        seed = stable_seed(
                            args.selection_seed, voxel_size,
                            configuration.configuration_id, test_run,
                            training_size, replicate)
                        candidates, candidate_ids = candidate_coverage_table(
                            pool=pool,
                            calibration=calibration,
                            training_size=training_size,
                            candidate_subsets=args.candidate_subsets,
                            seed=seed,
                        )
                        selected = select_coverage_quantiles(
                            candidates, coverage_targets,
                            metric="calibration_unique_coverage")
                        bank_directory = (
                            output_root / "cells" / size_slug(voxel_size) / "folds"
                            / configuration.configuration_id / test_run
                            / training_slug(training_size) / "rep_{:02d}".format(replicate))
                        bank_directory.mkdir(parents=True, exist_ok=True)
                        tier_by_hash = dict(zip(
                            selected["subset_sha256"].astype(str),
                            selected["coverage_tier"].astype(str)))
                        candidates.assign(
                            selected_tier=candidates["subset_sha256"].map(tier_by_hash).fillna("")
                        ).to_csv(
                            bank_directory / "candidate_coverages.csv.gz",
                            index=False, compression="gzip")
                        for row in selected.to_dict(orient="records"):
                            completed += 1
                            status = run_selected_cell(
                                pool=pool,
                                calibration=calibration,
                                testing=testing,
                                configuration=configuration,
                                test_run=test_run,
                                calibration_run=calibration_run,
                                grid=grid,
                                output_root=output_root,
                                signature=signature,
                                selected=row,
                                episode_ids=candidate_ids[str(row["subset_sha256"])],
                                replicate=replicate,
                                resume=args.resume,
                                save_models=args.save_models,
                                save_predictions=args.save_predictions,
                            )
                            print(
                                "[cell {}/{}] {} {} n={} {} calibration={:.3f} test={:.3f} auc-ready".format(
                                    completed, planned, configuration.configuration_id,
                                    test_run, training_size, row["coverage_tier"],
                                    status["calibration_unique_coverage"],
                                    status["test_unique_coverage"]),
                                flush=True,
                            )

    print("[aggregate] computing matched coverage summaries", flush=True)
    summary = aggregate_results(output_root)
    run_metadata.update({
        "status": "complete" if not args.smoke and summary["status"] == "complete" else summary["status"],
        "completed_at_utc": utc_now(),
        "elapsed_seconds": time.perf_counter() - started,
        "summary": summary,
    })
    atomic_json(existing_run, run_metadata)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
