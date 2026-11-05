#!/usr/bin/env python3
"""Run RQ2: compare VoxelSBFL with uncertainty and quality baselines.

The trajectory-level baselines do not require fitting. Each raw run is treated
as the same held-out test cell used by RQ1, and the completed RQ1 fold metrics
provide the frozen VoxelSBFL reference. Aggregation preserves the RQ1 hierarchy:
folds -> configurations -> VLA/condition blocks, with budgets equally weighted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from experiments.RQ2_baselines.analysis import (
        friedman_test,
        holm_adjust,
        paired_hierarchical_bootstrap,
        wilcoxon_signed_rank,
    )
    from experiments.RQ2_baselines.catalogue import METRIC_BY_ID, METRIC_SPECS
except ModuleNotFoundError:  # Direct execution with this directory on sys.path.
    from analysis import (  # type: ignore
        friedman_test,
        holm_adjust,
        paired_hierarchical_bootstrap,
        wilcoxon_signed_rank,
    )
    from catalogue import METRIC_BY_ID, METRIC_SPECS  # type: ignore

from vla_sbfl.metrics import average_precision, roc_auc
from experiments.RQ1.run_experiment import validate_primary_block_metrics


CONFIGURATION_PATTERN = re.compile(
    r"^(?P<condition>DH(?:_HybridDT|_HybridRF|_WeightedRF)?)_"
    r"(?P<budget>200|400)scenes$"
)
SCENE_PATTERN = re.compile(r"_Scenes_(?P<eval_id>\d+)$")
EXPECTED_CONDITIONS = ("DH", "DH_HybridDT", "DH_HybridRF", "DH_WeightedRF")
EXPECTED_BUDGETS = (200, 400)
EXPECTED_RUNS = tuple("run_" + str(index) for index in range(1, 11))
VLA_ORDER = ("EO1", "Groot", "Pi0", "SpatialVLA")
RAW_MODEL_ALIASES = {
    "EO1": ("EO1", "eo1"),
    "Groot": ("Groot", "GR00T", "Gr00t", "gr00t", "Isaac-GR00T"),
    "Pi0": ("Pi0", "PI0", "pi0"),
    "SpatialVLA": ("SpatialVLA", "Spatial_VLA", "spatialvla"),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_payload(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def git_revision() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_sort_key(run_id: str) -> Tuple[int, str]:
    match = re.search(r"(\d+)$", str(run_id))
    return (int(match.group(1)) if match else sys.maxsize, str(run_id))


def normalize_eval_id(value: object) -> str:
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0", text):
        return str(int(float(text)))
    return text


def resolve_raw_model_root(raw_root: Path, vla: str) -> Optional[Path]:
    root = raw_root.expanduser().resolve()
    if root.is_dir() and root.name in RAW_MODEL_ALIASES[vla]:
        return root
    for alias in RAW_MODEL_ALIASES[vla]:
        candidate = root / alias
        if candidate.is_dir():
            return candidate.resolve()
    return None


def discover_models(input_root: Path, raw_root: Path, requested: Sequence[str]) -> List[str]:
    invalid = sorted(set(requested) - set(VLA_ORDER))
    if invalid:
        raise ValueError("unknown VLA names: " + ", ".join(invalid))
    candidates = list(requested) if requested else list(VLA_ORDER)
    available = []
    for vla in candidates:
        if not (input_root / vla).is_dir():
            if requested:
                raise FileNotFoundError(input_root / vla)
            continue
        if resolve_raw_model_root(raw_root, vla) is None:
            if requested:
                raise FileNotFoundError(
                    "no raw model directory for {} under {}".format(vla, raw_root)
                )
            continue
        available.append(vla)
    if not available:
        raise ValueError("no VLA has both compact input and raw allMetrics data")
    return available


def configuration_id(vla: str, condition: str, budget: int) -> str:
    return "{}__{}__{}".format(vla, condition, budget)


def discover_configurations(
    input_root: Path,
    raw_root: Path,
    vlas: Sequence[str],
) -> List[dict]:
    configurations: List[dict] = []
    for vla in vlas:
        raw_model_root = resolve_raw_model_root(raw_root, vla)
        assert raw_model_root is not None
        for input_directory in sorted(path for path in (input_root / vla).iterdir() if path.is_dir()):
            match = CONFIGURATION_PATTERN.fullmatch(input_directory.name)
            if match is None:
                continue
            raw_directory = raw_model_root / input_directory.name
            if not raw_directory.is_dir():
                raise FileNotFoundError(raw_directory)
            condition = match.group("condition")
            budget = int(match.group("budget"))
            configurations.append({
                "configuration_id": configuration_id(vla, condition, budget),
                "vla": vla,
                "condition": condition,
                "budget": budget,
                "input_directory": input_directory.resolve(),
                "raw_directory": raw_directory.resolve(),
            })
    expected = {
        configuration_id(vla, condition, budget)
        for vla in vlas
        for condition in EXPECTED_CONDITIONS
        for budget in EXPECTED_BUDGETS
    }
    found = {item["configuration_id"] for item in configurations}
    if found != expected:
        raise ValueError(
            "incomplete design; missing={}, unexpected={}".format(
                sorted(expected - found), sorted(found - expected)
            )
        )
    return sorted(configurations, key=lambda item: item["configuration_id"])


def read_csv_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def numeric_leaves(value: object, path: str = "root") -> List[float]:
    if isinstance(value, bool):
        raise ValueError(path + " contains a Boolean")
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(path + " contains a non-finite number")
        return [number]
    if isinstance(value, list):
        result: List[float] = []
        for index, child in enumerate(value):
            result.extend(numeric_leaves(child, "{}[{}]".format(path, index)))
        return result
    if isinstance(value, dict):
        result = []
        for key in sorted(value):
            result.extend(numeric_leaves(value[key], "{}.{}".format(path, key)))
        return result
    if value is None:
        raise ValueError(path + " contains null")
    raise ValueError(path + " contains unsupported type " + type(value).__name__)


def load_metric_score(path: Path, direction: int) -> Tuple[Optional[float], int, str]:
    if not path.is_file():
        return None, 0, "missing_file"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = numeric_leaves(payload)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        return None, 0, "invalid:" + str(error)
    if not values:
        return None, 0, "empty"
    return float(direction * np.mean(np.asarray(values, dtype=np.float64))), len(values), "available"


def scene_map(results_directory: Path) -> Dict[str, Path]:
    if not results_directory.is_dir():
        raise FileNotFoundError(results_directory)
    result: Dict[str, Path] = {}
    for directory in results_directory.iterdir():
        if not directory.is_dir():
            continue
        match = SCENE_PATTERN.search(directory.name)
        if match is None:
            continue
        eval_id = normalize_eval_id(match.group("eval_id"))
        if eval_id in result:
            raise ValueError("duplicate scene directory for eval_id=" + eval_id)
        result[eval_id] = directory
    return result


def metrics_directory(scene_directory: Path) -> Path:
    candidates = sorted(
        path for path in scene_directory.rglob("allMetrics/0") if path.is_dir()
    )
    if len(candidates) != 1:
        raise ValueError(
            "{} contains {} allMetrics/0 directories".format(scene_directory, len(candidates))
        )
    return candidates[0]


def experiment_signature(protocol_path: Path) -> Tuple[str, dict]:
    runner_path = Path(__file__).resolve()
    details = {
        "protocol_sha256": sha256_file(protocol_path),
        "runner_sha256": sha256_file(runner_path),
        "analysis_sha256": sha256_file(runner_path.with_name("analysis.py")),
        "catalogue_sha256": sha256_file(runner_path.with_name("catalogue.py")),
        "metric_catalogue": [spec.to_dict() for spec in METRIC_SPECS],
        "trajectory_aggregation": "arithmetic mean over all finite scalar JSON leaves",
        "positive_class": "failure",
        "sbfl_reference": "harmonic_mean from frozen RQ1 folds",
    }
    return sha256_payload(details), details


def process_run(
    configuration: Mapping[str, object],
    run_id: str,
    output_root: Path,
    signature: str,
    resume: bool,
) -> dict:
    destination = output_root / "folds" / str(configuration["configuration_id"]) / run_id
    completion_path = destination / "fold.json"
    if completion_path.is_file():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("experiment_signature") != signature:
            raise ValueError("existing fold has a different signature: " + str(destination))
        if not resume:
            raise FileExistsError("fold exists and resume is disabled: " + str(destination))
        return completion

    input_run = Path(str(configuration["input_directory"])) / run_id
    raw_run = Path(str(configuration["raw_directory"])) / run_id
    episodes_path = input_run / "episodes.csv"
    evaluations_path = raw_run / "dh_evaluations.csv"
    episodes = read_csv_rows(episodes_path)
    evaluations = read_csv_rows(evaluations_path)
    if not episodes:
        raise ValueError("empty compact episode file: " + str(episodes_path))
    raw_by_eval = {normalize_eval_id(row["eval_id"]): row for row in evaluations}
    if len(raw_by_eval) != len(evaluations):
        raise ValueError("duplicate eval_id in " + str(evaluations_path))
    scenes = scene_map(raw_run / "results")
    if len(scenes) != len(episodes):
        raise ValueError("scene/episode count mismatch in " + str(raw_run))

    started = time.perf_counter()
    prediction_rows: List[dict] = []
    availability = {
        spec.metric_id: {"available": 0, "empty": 0, "missing_file": 0, "invalid": 0}
        for spec in METRIC_SPECS
    }
    for episode in episodes:
        eval_id = normalize_eval_id(episode["eval_id"])
        if eval_id not in raw_by_eval or eval_id not in scenes:
            raise ValueError("eval_id={} is not represented in all sources".format(eval_id))
        raw = raw_by_eval[eval_id]
        compact_failure = int(episode["failure"])
        if compact_failure != int(raw["failure"]):
            raise ValueError("failure-label mismatch for eval_id=" + eval_id)
        if str(episode.get("object_name", "")) != str(raw.get("object_name", "")):
            raise ValueError("object-name mismatch for eval_id=" + eval_id)
        metric_root = metrics_directory(scenes[eval_id])
        row = {
            "configuration_id": configuration["configuration_id"],
            "vla": configuration["vla"],
            "condition": configuration["condition"],
            "budget": configuration["budget"],
            "test_run": run_id,
            "eval_id": eval_id,
            "episode_id": episode["episode_id"],
            "failure": compact_failure,
            "object_name": episode.get("object_name", ""),
        }
        for spec in METRIC_SPECS:
            score, scalar_count, status = load_metric_score(
                metric_root / spec.filename, spec.direction
            )
            row["score__" + spec.metric_id] = score
            row["scalar_count__" + spec.metric_id] = scalar_count
            status_key = "invalid" if status.startswith("invalid:") else status
            availability[spec.metric_id][status_key] += 1
        prediction_rows.append(row)

    predictions = pd.DataFrame(prediction_rows).sort_values("eval_id", kind="mergesort")
    labels = predictions["failure"].to_numpy(dtype=np.int8)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("both outcome classes are required in " + str(input_run))

    metric_rows = []
    for spec in METRIC_SPECS:
        scores = predictions["score__" + spec.metric_id]
        nonmissing = int(scores.notna().sum())
        # Partial availability is not evaluated because dropping episodes would
        # change the test set and could introduce outcome-dependent selection.
        if nonmissing != len(predictions):
            continue
        values = scores.to_numpy(dtype=np.float64)
        metric_rows.append({
            "configuration_id": configuration["configuration_id"],
            "vla": configuration["vla"],
            "condition": configuration["condition"],
            "budget": configuration["budget"],
            "test_run": run_id,
            "method_id": spec.metric_id,
            "method": spec.name,
            "family": spec.family,
            "source": "external_baseline",
            "task_aware": spec.task_aware,
            "roc_auc": roc_auc(labels, values),
            "average_precision": average_precision(labels, values),
            "episodes": len(labels),
            "failures": int(labels.sum()),
            "failure_prevalence": float(labels.mean()),
            "distinct_scores": int(np.unique(values).size),
        })

    destination.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(
        destination / "predictions.csv.gz", index=False, compression="gzip"
    )
    pd.DataFrame(metric_rows).to_csv(destination / "metrics.csv", index=False)
    availability_rows = []
    for spec in METRIC_SPECS:
        availability_rows.append({
            "configuration_id": configuration["configuration_id"],
            "vla": configuration["vla"],
            "condition": configuration["condition"],
            "budget": configuration["budget"],
            "test_run": run_id,
            "method_id": spec.metric_id,
            "method": spec.name,
            "episodes": len(predictions),
            **availability[spec.metric_id],
        })
    pd.DataFrame(availability_rows).to_csv(destination / "availability.csv", index=False)
    completion = {
        "status": "complete",
        "completed_at_utc": utc_now(),
        "experiment_signature": signature,
        "configuration_id": configuration["configuration_id"],
        "vla": configuration["vla"],
        "condition": configuration["condition"],
        "budget": configuration["budget"],
        "test_run": run_id,
        "episodes": len(predictions),
        "failures": int(labels.sum()),
        "passes": int((labels == 0).sum()),
        "complete_metrics": len(metric_rows),
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(completion_path, completion)
    return completion


def reshape_paired_folds(
    folds: pd.DataFrame,
    method_id: str,
    block_order: Sequence[str],
) -> np.ndarray:
    subset = folds[folds["method_id"] == method_id].copy()
    budgets = sorted(int(value) for value in subset["budget"].unique())
    runs = sorted(subset["test_run"].astype(str).unique(), key=run_sort_key)
    pivot = subset.pivot(
        index=["primary_block_id", "budget", "test_run"],
        columns="method_id",
        values="roc_auc",
    )
    result = np.empty((len(block_order), len(budgets), len(runs)), dtype=np.float64)
    for block_index, block in enumerate(block_order):
        for budget_index, budget in enumerate(budgets):
            values = pivot.loc[(block, budget)].reindex(runs)[method_id]
            if values.isna().any():
                raise ValueError("incomplete fold grid for " + method_id)
            result[block_index, budget_index] = values.to_numpy(dtype=np.float64)
    return result


def aggregate_results(
    output_root: Path,
    rq1_fold_metrics_path: Path,
    selected_vlas: Sequence[str],
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> dict:
    aggregate_directory = output_root / "aggregate"
    aggregate_directory.mkdir(parents=True, exist_ok=True)
    metric_paths = sorted((output_root / "folds").glob("*/*/metrics.csv"))
    availability_paths = sorted((output_root / "folds").glob("*/*/availability.csv"))
    status_paths = sorted((output_root / "folds").glob("*/*/fold.json"))
    if not metric_paths:
        raise ValueError("no completed baseline fold metrics were found")
    baseline_folds = pd.concat([pd.read_csv(path) for path in metric_paths], ignore_index=True)
    availability = pd.concat(
        [pd.read_csv(path) for path in availability_paths], ignore_index=True
    )
    statuses = pd.DataFrame(
        [json.loads(path.read_text(encoding="utf-8")) for path in status_paths]
    )
    if not rq1_fold_metrics_path.is_file():
        raise FileNotFoundError(rq1_fold_metrics_path)
    rq1 = pd.read_csv(rq1_fold_metrics_path)
    required = {
        "configuration_id", "vla", "condition", "budget", "test_run",
        "formula_id", "technique", "roc_auc", "average_precision", "episodes",
        "failures", "failure_prevalence", "distinct_scores",
    }
    missing = required - set(rq1.columns)
    if missing:
        raise ValueError("RQ1 fold metrics are missing: " + ", ".join(sorted(missing)))
    rq1 = rq1[
        (rq1["formula_id"] == "harmonic_mean")
        & rq1["vla"].astype(str).isin(selected_vlas)
    ].copy()
    expected_configurations = set(statuses["configuration_id"].astype(str))
    rq1 = rq1[rq1["configuration_id"].astype(str).isin(expected_configurations)]
    if len(rq1) != len(statuses):
        raise ValueError(
            "RQ1 reference does not contain exactly one Harmonic Mean row per selected fold"
        )
    rq1 = rq1.rename(columns={"formula_id": "method_id", "technique": "method"})
    rq1["family"] = "VoxelSBFL"
    rq1["source"] = "rq1_reference"
    rq1["task_aware"] = False
    shared_columns = [
        "configuration_id", "vla", "condition", "budget", "test_run",
        "method_id", "method", "family", "source", "task_aware", "roc_auc",
        "average_precision", "episodes", "failures", "failure_prevalence",
        "distinct_scores",
    ]
    folds = pd.concat(
        [rq1[shared_columns], baseline_folds[shared_columns]], ignore_index=True
    )
    folds["primary_block_id"] = (
        folds["vla"].astype(str) + "__" + folds["condition"].astype(str)
    )
    duplicate_keys = ["configuration_id", "test_run", "method_id"]
    if folds.duplicated(duplicate_keys).any():
        raise ValueError("duplicate method rows in the combined fold data")
    folds.to_csv(
        aggregate_directory / "fold_metrics.csv.gz", index=False, compression="gzip"
    )
    availability.to_csv(aggregate_directory / "metric_availability_by_fold.csv", index=False)
    statuses.sort_values(["configuration_id", "test_run"]).to_csv(
        aggregate_directory / "fold_status.csv", index=False
    )

    availability_summary = availability.groupby(
        ["vla", "method_id", "method"], as_index=False
    ).agg(
        episodes=("episodes", "sum"),
        available=("available", "sum"),
        empty=("empty", "sum"),
        missing_file=("missing_file", "sum"),
        invalid=("invalid", "sum"),
        folds=("test_run", "size"),
    )
    availability_summary["availability_fraction"] = (
        availability_summary["available"] / availability_summary["episodes"]
    )
    availability_summary.to_csv(
        aggregate_directory / "metric_availability.csv", index=False
    )

    configurations = folds.groupby(
        ["configuration_id", "vla", "condition", "budget", "method_id", "method",
         "family", "source", "task_aware"],
        as_index=False,
    ).agg(
        roc_auc=("roc_auc", "mean"),
        average_precision=("average_precision", "mean"),
        failure_prevalence=("failure_prevalence", "mean"),
        folds=("test_run", "nunique"),
    )
    configurations.to_csv(aggregate_directory / "configuration_metrics.csv", index=False)

    primary = configurations.groupby(
        ["vla", "condition", "method_id", "method", "family", "source", "task_aware"],
        as_index=False,
    ).agg(
        roc_auc=("roc_auc", "mean"),
        average_precision=("average_precision", "mean"),
        budgets=("budget", "nunique"),
        configurations=("configuration_id", "nunique"),
    )
    primary["primary_block_id"] = (
        primary["vla"].astype(str) + "__" + primary["condition"].astype(str)
    )
    primary["roc_auc_rank"] = primary.groupby("primary_block_id")["roc_auc"].rank(
        method="average", ascending=False
    )
    primary.to_csv(aggregate_directory / "primary_block_metrics.csv", index=False)

    by_vla = primary.groupby(
        ["vla", "method_id", "method", "family", "source", "task_aware"],
        as_index=False,
    ).agg(
        macro_roc_auc=("roc_auc", "mean"),
        macro_average_precision=("average_precision", "mean"),
        mean_rank=("roc_auc_rank", "mean"),
        primary_blocks=("primary_block_id", "nunique"),
    )
    by_vla["within_vla_rank"] = by_vla.groupby("vla")["macro_roc_auc"].rank(
        method="min", ascending=False
    ).astype(int)
    by_vla.to_csv(aggregate_directory / "performance_by_vla.csv", index=False)

    expected_blocks = {
        "{}__{}".format(vla, condition)
        for vla in selected_vlas
        for condition in EXPECTED_CONDITIONS
    }
    method_block_counts = primary.groupby("method_id")["primary_block_id"].nunique()
    complete_methods = sorted(
        method_id
        for method_id, count in method_block_counts.items()
        if int(count) == len(expected_blocks)
        and set(primary.loc[primary["method_id"] == method_id, "primary_block_id"]) == expected_blocks
    )
    if "harmonic_mean" not in complete_methods:
        raise ValueError("Harmonic Mean is not complete across the selected design")
    overall = primary.groupby(
        ["method_id", "method", "family", "source", "task_aware"], as_index=False
    ).agg(
        macro_roc_auc=("roc_auc", "mean"),
        macro_average_precision=("average_precision", "mean"),
        mean_rank=("roc_auc_rank", "mean"),
        primary_blocks=("primary_block_id", "nunique"),
        vlas=("vla", "nunique"),
    )
    overall["complete_primary_comparison"] = overall["method_id"].isin(complete_methods)
    complete_ranks = overall.loc[overall["complete_primary_comparison"], "macro_roc_auc"].rank(
        method="min", ascending=False
    )
    overall["overall_rank"] = np.nan
    overall.loc[overall["complete_primary_comparison"], "overall_rank"] = complete_ranks
    overall = overall.sort_values(
        ["complete_primary_comparison", "macro_roc_auc", "method_id"],
        ascending=[False, False, True],
    )
    overall.to_csv(aggregate_directory / "overall_performance.csv", index=False)

    block_order = sorted(expected_blocks)
    complete_primary = primary[primary["method_id"].isin(complete_methods)]
    pivot = complete_primary.pivot(
        index="primary_block_id", columns="method_id", values="roc_auc"
    ).reindex(index=block_order, columns=complete_methods)
    if pivot.isna().any().any():
        raise ValueError("complete primary comparison contains missing cells")
    friedman = friedman_test(pivot.to_numpy(dtype=np.float64))
    reference_blocks = pivot["harmonic_mean"].to_numpy(dtype=np.float64)
    raw_p_values: Dict[str, float] = {}
    comparisons: Dict[str, dict] = {}
    fold_block_order = sorted(expected_blocks)
    reference_folds = reshape_paired_folds(folds, "harmonic_mean", fold_block_order)
    for offset, method_id in enumerate(complete_methods):
        if method_id == "harmonic_mean":
            continue
        candidate_blocks = pivot[method_id].to_numpy(dtype=np.float64)
        wilcoxon = wilcoxon_signed_rank(candidate_blocks, reference_blocks)
        raw_p_values[method_id] = wilcoxon.p_value
        candidate_folds = reshape_paired_folds(folds, method_id, fold_block_order)
        ref_low, ref_high, gap_low, gap_high = paired_hierarchical_bootstrap(
            reference_folds,
            candidate_folds,
            iterations=bootstrap_iterations,
            seed=bootstrap_seed + offset,
        )
        comparisons[method_id] = {
            "candidate_macro_roc_auc": float(candidate_blocks.mean()),
            "reference_macro_roc_auc": float(reference_blocks.mean()),
            "auc_gap_candidate_minus_reference": float(
                (candidate_blocks - reference_blocks).mean()
            ),
            "gap_ci_low": gap_low,
            "gap_ci_high": gap_high,
            "reference_ci_low": ref_low,
            "reference_ci_high": ref_high,
            "raw_p": wilcoxon.p_value,
            "rank_biserial": wilcoxon.rank_biserial,
            "nonzero_pairs": wilcoxon.nonzero_pairs,
        }
    adjusted = holm_adjust(raw_p_values)
    comparison_rows = []
    method_metadata = primary.drop_duplicates("method_id").set_index("method_id")
    for method_id, values in comparisons.items():
        gap = float(values["auc_gap_candidate_minus_reference"])
        comparison_rows.append({
            "method_id": method_id,
            "method": method_metadata.loc[method_id, "method"],
            "family": method_metadata.loc[method_id, "family"],
            **values,
            "holm_p": adjusted[method_id],
            "practically_equivalent": bool(
                abs(gap) < 0.01 and adjusted[method_id] >= 0.05
            ),
        })
    comparisons_frame = pd.DataFrame(comparison_rows).sort_values(
        ["candidate_macro_roc_auc", "method_id"], ascending=[False, True]
    )
    comparisons_frame.to_csv(
        aggregate_directory / "statistical_comparison.csv", index=False
    )
    statistical_tests = {
        "friedman": {
            "statistic": friedman.statistic,
            "degrees_of_freedom": friedman.degrees_of_freedom,
            "p_value": friedman.p_value,
            "primary_blocks": friedman.blocks,
            "methods": friedman.techniques,
        },
        "post_hoc_reference": "harmonic_mean",
        "holm_family": sorted(raw_p_values),
        "bootstrap_iterations": bootstrap_iterations,
        "bootstrap_seed": bootstrap_seed,
        "practical_equivalence_margin": 0.01,
    }
    atomic_json(aggregate_directory / "statistical_tests.json", statistical_tests)
    summary = {
        "status": "complete",
        "selected_vlas": list(selected_vlas),
        "folds": int(statuses.shape[0]),
        "episodes": int(statuses["episodes"].sum()),
        "available_external_methods": int(baseline_folds["method_id"].nunique()),
        "complete_comparison_methods_including_reference": len(complete_methods),
        "complete_method_ids": complete_methods,
        "primary_blocks": len(expected_blocks),
        "reference_macro_roc_auc": float(reference_blocks.mean()),
        "best_complete_method": str(overall.iloc[0]["method"]),
        "best_complete_macro_roc_auc": float(overall.iloc[0]["macro_roc_auc"]),
    }
    atomic_json(aggregate_directory / "summary.json", summary)
    return summary


def regenerate_statistical_review(
    aggregate_directory: Path,
    method: str = "exact",
) -> dict:
    """Recompute the eight paired baseline contrasts across 16 VLA-strategy blocks.

    Holm covers all eight comparisons. Practical equivalence is a descriptive
    mean-gap/non-significance rule, not a formal equivalence test. Retained
    bootstrap intervals are pointwise and not adjusted for Holm or selection.
    """

    if method not in {"approximate", "exact"}:
        raise ValueError("Wilcoxon method must be 'approximate' or 'exact'")
    aggregate_directory = aggregate_directory.expanduser().resolve()
    primary_path = aggregate_directory / "primary_block_metrics.csv"
    historical_path = aggregate_directory / "statistical_comparison.csv"
    if not primary_path.is_file() or not historical_path.is_file():
        raise FileNotFoundError("final baseline aggregate inputs are incomplete")
    primary = pd.read_csv(primary_path)
    historical = pd.read_csv(historical_path)
    expected_ids = (
        "action_acceleration_instability", "action_position_instability",
        "action_velocity_instability", "optimal_trajectory",
        "tcp_acceleration_instability", "tcp_position_instability",
        "tcp_velocity_instability", "trajectory_instability",
    )
    names = {key: METRIC_BY_ID[key].name for key in expected_ids}
    names["harmonic_mean"] = "Harmonic Mean"
    validate_primary_block_metrics(primary.rename(columns={
        "method_id": "formula_id", "method": "technique"}), names)
    required = {"primary_block_id", "method_id", "method", "family", "roc_auc"}
    if required - set(primary.columns):
        raise ValueError("baseline primary block metrics have an incompatible schema")
    if primary["primary_block_id"].nunique() != 16:
        raise ValueError("the final baseline review requires 16 primary blocks")
    methods = sorted(primary["method_id"].astype(str).unique())
    if len(methods) != 9 or "harmonic_mean" not in methods:
        raise ValueError("the final baseline review requires Harmonic Mean plus eight baselines")
    pivot = primary.pivot(
        index="primary_block_id", columns="method_id", values="roc_auc"
    ).reindex(columns=methods)
    if pivot.shape != (16, 9) or pivot.isna().any().any():
        raise ValueError("the final baseline paired design is incomplete")
    historical = historical.set_index("method_id", drop=False)
    candidates = [method_id for method_id in methods if method_id != "harmonic_mean"]
    if set(historical.index.astype(str)) != set(candidates):
        raise ValueError("historical baseline comparisons do not match the final design")

    reference = pivot["harmonic_mean"].to_numpy(dtype=np.float64)
    tests = {}
    raw = {}
    for method_id in candidates:
        result = wilcoxon_signed_rank(
            pivot[method_id].to_numpy(dtype=np.float64), reference, method=method)
        tests[method_id] = result
        raw[method_id] = result.p_value
    adjusted = holm_adjust(raw)
    rows = []
    for method_id in candidates:
        old = historical.loc[method_id].to_dict()
        candidate_mean = float(pivot[method_id].mean())
        reference_mean = float(reference.mean())
        if not math.isclose(
            candidate_mean, float(old["candidate_macro_roc_auc"]), rel_tol=0.0, abs_tol=1e-15
        ) or not math.isclose(
            reference_mean, float(old["reference_macro_roc_auc"]), rel_tol=0.0, abs_tol=1e-15
        ):
            raise ValueError("historical baseline ROC-AUC values do not reconcile")
        result = tests[method_id]
        old.update({
            "raw_p": result.p_value,
            "rank_biserial": result.rank_biserial,
            "nonzero_pairs": result.nonzero_pairs,
            "holm_p": adjusted[method_id],
            "practically_equivalent": bool(
                abs(float(old["auc_gap_candidate_minus_reference"])) < 0.01
                and adjusted[method_id] >= 0.05),
            "wilcoxon_method": method,
            "holm_family_size": len(candidates),
            "significant_0p05": adjusted[method_id] < 0.05,
        })
        rows.append(old)
    revised = pd.DataFrame(rows).sort_values(
        ["candidate_macro_roc_auc", "method_id"], ascending=[False, True])
    output_path = aggregate_directory / "statistical_comparison_{}.csv".format(method)
    revised.to_csv(output_path, index=False)
    payload = {
        "status": "complete",
        "wilcoxon_method": method,
        "wilcoxon_alternative": "two-sided",
        "post_hoc_reference": "harmonic_mean",
        "holm_family_size": len(candidates),
        "primary_blocks": 16,
        "comparison_count": len(revised),
        "significant_comparisons_0p05": int(revised["significant_0p05"].sum()),
        "source_sha256": {
            primary_path.name: sha256_file(primary_path),
            historical_path.name: sha256_file(historical_path),
        },
        "output": output_path.name,
        "output_sha256": sha256_file(output_path),
        "z_score_reported": False,
    }
    atomic_json(
        aggregate_directory / "statistical_review_{}.json".format(method), payload)
    return payload


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=root / "input")
    parser.add_argument("--raw-root", type=Path, default=root / "VLA_SBFL_EO1" / "Data")
    parser.add_argument(
        "--rq1-fold-metrics",
        type=Path,
        default=root / "experiments" / "RQ1" / "results" / "phase_a_primary"
        / "aggregate" / "fold_metrics.csv.gz",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "experiments" / "RQ2_baselines" / "results" / "baseline_comparison",
    )
    parser.add_argument("--vla", action="append", default=[])
    parser.add_argument("--configuration", action="append", default=[])
    parser.add_argument("--max-configurations", type=int)
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261025)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Rebuild aggregate statistics from already completed fold artifacts.",
    )
    parser.add_argument(
        "--statistical-review-only",
        action="store_true",
        help="Recompute only method-labelled statistics from aggregate block metrics.",
    )
    parser.add_argument(
        "--wilcoxon-method", choices=("approximate", "exact"), default="exact")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_configurations is not None and args.max_configurations <= 0:
        raise ValueError("--max-configurations must be positive")
    if args.max_runs is not None and args.max_runs <= 0:
        raise ValueError("--max-runs must be positive")
    if args.bootstrap_iterations < 100:
        raise ValueError("--bootstrap-iterations must be at least 100")
    selected_modes = sum((args.extract_only, args.analysis_only, args.statistical_review_only))
    if selected_modes > 1:
        raise ValueError("extraction, aggregation, and statistical review modes are exclusive")
    input_root = args.input_root.expanduser().resolve()
    raw_root = args.raw_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if args.statistical_review_only:
        if not output_root.is_dir():
            raise FileNotFoundError(output_root)
        summary = regenerate_statistical_review(
            output_root / "aggregate", method=args.wilcoxon_method)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return 0
    output_root.mkdir(parents=True, exist_ok=True)
    vlas = discover_models(input_root, raw_root, args.vla)
    configurations = discover_configurations(input_root, raw_root, vlas)
    if args.configuration:
        requested = set(args.configuration)
        known = {str(item["configuration_id"]) for item in configurations}
        if requested - known:
            raise ValueError("unknown configurations: " + ", ".join(sorted(requested - known)))
        configurations = [
            item for item in configurations if item["configuration_id"] in requested
        ]
    if args.max_configurations is not None:
        configurations = configurations[: args.max_configurations]
    # Subsetting configurations is for smoke tests/extraction only; aggregate
    # inference requires the complete factorial design for every selected VLA.
    full_design = len(configurations) == len(vlas) * len(EXPECTED_CONDITIONS) * len(EXPECTED_BUDGETS)
    protocol_path = Path(__file__).resolve().with_name("RQ2.md")
    signature, signature_details = experiment_signature(protocol_path)
    run_metadata = {
        "status": "running",
        "started_at_utc": utc_now(),
        "experiment_signature": signature,
        "signature_details": signature_details,
        "input_root": str(input_root),
        "raw_root": str(raw_root),
        "rq1_fold_metrics": str(args.rq1_fold_metrics.expanduser().resolve()),
        "output_root": str(output_root),
        "vlas": vlas,
        "configurations": [item["configuration_id"] for item in configurations],
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "git_revision_at_launch": git_revision(),
        "bootstrap_iterations": args.bootstrap_iterations,
        "bootstrap_seed": args.bootstrap_seed,
        "analysis_only": args.analysis_only,
    }
    atomic_json(output_root / "run.json", run_metadata)
    if args.analysis_only:
        if not full_design:
            raise ValueError("--analysis-only requires the complete selected design")
        summary = aggregate_results(
            output_root=output_root,
            rq1_fold_metrics_path=args.rq1_fold_metrics.expanduser().resolve(),
            selected_vlas=vlas,
            bootstrap_iterations=args.bootstrap_iterations,
            bootstrap_seed=args.bootstrap_seed,
        )
        run_metadata.update({
            "status": "complete",
            "completed_at_utc": utc_now(),
            "summary": summary,
        })
        atomic_json(output_root / "run.json", run_metadata)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return 0
    total = len(configurations) * (
        min(10, args.max_runs) if args.max_runs is not None else 10
    )
    completed = 0
    started = time.perf_counter()
    for config_index, configuration in enumerate(configurations, start=1):
        input_runs = sorted(
            (
                path.name
                for path in Path(str(configuration["input_directory"])).iterdir()
                if path.is_dir() and path.name.startswith("run_")
            ),
            key=run_sort_key,
        )
        raw_runs = sorted(
            (
                path.name
                for path in Path(str(configuration["raw_directory"])).iterdir()
                if path.is_dir() and path.name.startswith("run_")
            ),
            key=run_sort_key,
        )
        if tuple(input_runs) != EXPECTED_RUNS or tuple(raw_runs) != EXPECTED_RUNS:
            raise ValueError("expected run_1 through run_10 for " + str(configuration["configuration_id"]))
        runs = input_runs[: args.max_runs] if args.max_runs is not None else input_runs
        for run_id in runs:
            completed += 1
            print(
                "[fold {}/{}; config {}/{}] {} test={}".format(
                    completed, total, config_index, len(configurations),
                    configuration["configuration_id"], run_id,
                ),
                flush=True,
            )
            status = process_run(
                configuration, run_id, output_root, signature, args.resume
            )
            print(
                "  episodes={} complete_metrics={} time={:.2f}s".format(
                    status["episodes"], status["complete_metrics"], status["elapsed_seconds"]
                ),
                flush=True,
            )
    if args.extract_only or not full_design or args.max_runs is not None:
        run_metadata.update({
            "status": "extraction_complete",
            "completed_at_utc": utc_now(),
            "elapsed_seconds": time.perf_counter() - started,
            "aggregate_skipped": True,
            "aggregate_skip_reason": "extract-only or incomplete smoke-test design",
        })
        atomic_json(output_root / "run.json", run_metadata)
        return 0
    print("[aggregate] combining baselines with the frozen RQ1 reference", flush=True)
    summary = aggregate_results(
        output_root=output_root,
        rq1_fold_metrics_path=args.rq1_fold_metrics.expanduser().resolve(),
        selected_vlas=vlas,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
    )
    run_metadata.update({
        "status": "complete",
        "completed_at_utc": utc_now(),
        "elapsed_seconds": time.perf_counter() - started,
        "summary": summary,
    })
    atomic_json(output_root / "run.json", run_metadata)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
