#!/usr/bin/env python3
"""Run the frozen RQ1 Phase A leave-one-run-out experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from experiments.RQ1.analysis import (
        friedman_test,
        hierarchical_budget_bootstrap,
        holm_adjust,
        wilcoxon_signed_rank,
    )
except ModuleNotFoundError:  # Direct execution places this directory on sys.path.
    from analysis import (  # type: ignore
        friedman_test,
        hierarchical_budget_bootstrap,
        holm_adjust,
        wilcoxon_signed_rank,
    )
from vla_sbfl.dataset import (
    TrajectoryDataset,
    discover_run_directories,
    load_trajectory_dataset,
)
from vla_sbfl.formulas import FORMULA_REGISTRY, TABLE4_FORMULA_IDS
from vla_sbfl.metrics import evaluate_predictions
from vla_sbfl.model import SBFLModel, episode_id_hash
from vla_sbfl.scoring import TrajectoryScorer
from vla_sbfl.voxelizer import VoxelGrid


CONFIGURATION_PATTERN = re.compile(
    r"^(?P<condition>DH(?:_HybridDT|_HybridRF|_WeightedRF)?)_"
    r"(?P<budget>200|400)scenes$"
)
EXPECTED_VLAS = ("EO1", "Groot", "Pi0", "SpatialVLA")
EXPECTED_CONDITIONS = ("DH", "DH_HybridDT", "DH_HybridRF", "DH_WeightedRF")
EXPECTED_BUDGETS = (200, 400)
EXPECTED_RUNS = tuple("run_" + str(index) for index in range(1, 11))


@dataclass(frozen=True)
class Configuration:
    vla: str
    condition: str
    budget: int
    directory: Path

    @property
    def configuration_id(self) -> str:
        return "{}__{}__{}".format(self.vla, self.condition, self.budget)

    @property
    def algorithm(self) -> str:
        return self.directory.name


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_payload(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def run_sort_key(run_id: str) -> Tuple[int, str]:
    match = re.search(r"(\d+)$", str(run_id))
    return (int(match.group(1)) if match else sys.maxsize, str(run_id))


def discover_configurations(input_root: Path) -> Tuple[Configuration, ...]:
    root = input_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    configurations = []
    for vla_directory in sorted(path for path in root.iterdir() if path.is_dir()):
        for configuration_directory in sorted(
            path for path in vla_directory.iterdir() if path.is_dir()
        ):
            match = CONFIGURATION_PATTERN.fullmatch(configuration_directory.name)
            if match is None:
                continue
            configurations.append(Configuration(
                vla=vla_directory.name,
                condition=match.group("condition"),
                budget=int(match.group("budget")),
                directory=configuration_directory.resolve(),
            ))
    identities = {configuration.configuration_id for configuration in configurations}
    expected = {
        "{}__{}__{}".format(vla, condition, budget)
        for vla in EXPECTED_VLAS
        for condition in EXPECTED_CONDITIONS
        for budget in EXPECTED_BUDGETS
    }
    missing = sorted(expected - identities)
    unexpected = sorted(identities - expected)
    if missing or unexpected or len(configurations) != 32:
        raise ValueError(
            "expected the frozen 32 configurations; missing={}, unexpected={}".format(
                missing, unexpected))
    return tuple(sorted(
        configurations,
        key=lambda item: (item.vla, item.condition, item.budget),
    ))


def subset_dataset(dataset: TrajectoryDataset, episode_ids: Iterable[str]) -> TrajectoryDataset:
    wanted = {str(value) for value in episode_ids}
    episodes = dataset.episodes[
        dataset.episodes["episode_id"].astype(str).isin(wanted)
    ].copy()
    if len(episodes) != len(wanted):
        raise ValueError("one or more selected episode IDs are missing")
    visits = {episode_id: dataset.visits[episode_id] for episode_id in wanted}
    return TrajectoryDataset(episodes=episodes.reset_index(drop=True), visits=visits)


def trajectory_hashes(run_directories: Sequence[Path]) -> Tuple[Dict[str, str], dict]:
    hashes: Dict[str, str] = {}
    coordinate_minimum = np.full(3, np.inf, dtype=np.float64)
    coordinate_maximum = np.full(3, -np.inf, dtype=np.float64)
    duplicate_steps = 0
    rows = 0
    for run_directory in run_directories:
        trajectory_path = run_directory / "trajectories.csv"
        frame = pd.read_csv(
            trajectory_path,
            dtype={"episode_id": str},
            usecols=["episode_id", "step", "x", "y", "z"],
            encoding="utf-8-sig",
        )
        if frame.empty:
            raise ValueError("empty trajectory file: " + str(trajectory_path))
        numeric = frame[["step", "x", "y", "z"]].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
            raise ValueError("non-finite trajectory value: " + str(trajectory_path))
        duplicate_steps += int(frame.duplicated(["episode_id", "step"]).sum())
        xyz = numeric[["x", "y", "z"]].to_numpy(dtype=np.float64)
        coordinate_minimum = np.minimum(coordinate_minimum, xyz.min(axis=0))
        coordinate_maximum = np.maximum(coordinate_maximum, xyz.max(axis=0))
        rows += len(frame)
        frame = frame.assign(
            step=numeric["step"].astype(np.int64),
            x=numeric["x"].astype(np.float64),
            y=numeric["y"].astype(np.float64),
            z=numeric["z"].astype(np.float64),
        ).sort_values(["episode_id", "step"], kind="mergesort")
        for episode_id, group in frame.groupby("episode_id", sort=False):
            digest = hashlib.sha256()
            digest.update(group["step"].to_numpy(dtype="<i8").tobytes())
            digest.update(group[["x", "y", "z"]].to_numpy(dtype="<f8").tobytes())
            hashes[str(episode_id)] = digest.hexdigest()
    return hashes, {
        "trajectory_rows": rows,
        "duplicate_episode_steps": duplicate_steps,
        "x_min": float(coordinate_minimum[0]),
        "x_max": float(coordinate_maximum[0]),
        "y_min": float(coordinate_minimum[1]),
        "y_max": float(coordinate_maximum[1]),
        "z_min": float(coordinate_minimum[2]),
        "z_max": float(coordinate_maximum[2]),
    }


def validate_configuration_metadata(
    configuration: Configuration,
    episodes: pd.DataFrame,
) -> Tuple[str, ...]:
    required = {
        "model", "algorithm", "budget", "run_id", "eval_id", "episode_id",
        "failure", "object_name", "num_steps",
    }
    missing = required - set(episodes.columns)
    if missing:
        raise ValueError(
            configuration.configuration_id + " missing columns: " + ", ".join(sorted(missing)))
    if set(episodes["model"].astype(str)) != {configuration.vla}:
        raise ValueError(configuration.configuration_id + " has inconsistent model values")
    if set(episodes["algorithm"].astype(str)) != {configuration.algorithm}:
        raise ValueError(configuration.configuration_id + " has inconsistent algorithm values")
    if set(pd.to_numeric(episodes["budget"], errors="coerce")) != {configuration.budget}:
        raise ValueError(configuration.configuration_id + " has inconsistent budget values")
    runs = tuple(sorted(episodes["run_id"].astype(str).unique(), key=run_sort_key))
    if runs != EXPECTED_RUNS:
        raise ValueError(
            "{} expected runs {}, found {}".format(
                configuration.configuration_id, EXPECTED_RUNS, runs))
    for run_id, frame in episodes.groupby(episodes["run_id"].astype(str)):
        if frame["eval_id"].duplicated().any():
            raise ValueError(configuration.configuration_id + " duplicate eval_id in " + run_id)
        labels = set(frame["failure"].astype(int))
        if labels != {0, 1}:
            raise ValueError(
                "{} {} does not contain both outcome classes".format(
                    configuration.configuration_id, run_id))
    return runs


def audit_all(
    configurations: Sequence[Configuration],
    grid: VoxelGrid,
    audit_directory: Path,
    chunk_size: int,
) -> dict:
    audit_directory.mkdir(parents=True, exist_ok=True)
    configuration_rows: List[dict] = []
    fold_rows: List[dict] = []
    episode_frames: List[pd.DataFrame] = []
    started = time.perf_counter()
    for index, configuration in enumerate(configurations, start=1):
        print(
            "[audit {}/{}] {}".format(
                index, len(configurations), configuration.configuration_id),
            flush=True,
        )
        run_directories = discover_run_directories([configuration.directory])
        dataset = load_trajectory_dataset(
            [configuration.directory], grid=grid, require_labels=True,
            chunk_size=chunk_size)
        episodes = dataset.episodes.copy()
        runs = validate_configuration_metadata(configuration, episodes)
        raw_hashes, coordinates = trajectory_hashes(run_directories)
        if coordinates["duplicate_episode_steps"]:
            raise ValueError(
                configuration.configuration_id + " contains duplicate episode/step rows")
        if set(raw_hashes) != set(episodes["episode_id"].astype(str)):
            raise ValueError(configuration.configuration_id + " trajectory hash coverage mismatch")
        failures = int(episodes["failure"].sum())
        configuration_rows.append({
            "configuration_id": configuration.configuration_id,
            "vla": configuration.vla,
            "condition": configuration.condition,
            "budget": configuration.budget,
            "algorithm": configuration.algorithm,
            "input_directory": str(configuration.directory),
            "runs": len(runs),
            "episodes": len(episodes),
            "failures": failures,
            "passes": len(episodes) - failures,
            **coordinates,
        })
        manifest_columns = [
            "episode_id", "run_id", "eval_id", "failure", "object_name", "num_steps"]
        manifest = episodes[manifest_columns].copy()
        manifest.insert(0, "configuration_id", configuration.configuration_id)
        manifest.insert(1, "vla", configuration.vla)
        manifest.insert(2, "condition", configuration.condition)
        manifest.insert(3, "budget", configuration.budget)
        manifest["trajectory_sha256"] = manifest["episode_id"].map(raw_hashes)
        episode_frames.append(manifest)
        for test_run in runs:
            test = episodes[episodes["run_id"].astype(str) == test_run]
            training = episodes[episodes["run_id"].astype(str) != test_run]
            train_ids = tuple(training["episode_id"].astype(str))
            test_ids = tuple(test["episode_id"].astype(str))
            if set(train_ids) & set(test_ids):
                raise ValueError("train/test overlap in " + configuration.configuration_id)
            fold_rows.append({
                "configuration_id": configuration.configuration_id,
                "vla": configuration.vla,
                "condition": configuration.condition,
                "budget": configuration.budget,
                "test_run": test_run,
                "training_episodes": len(training),
                "training_failures": int(training["failure"].sum()),
                "training_passes": int((training["failure"] == 0).sum()),
                "test_episodes": len(test),
                "test_failures": int(test["failure"].sum()),
                "test_passes": int((test["failure"] == 0).sum()),
                "training_episode_ids_sha256": episode_id_hash(train_ids),
                "test_episode_ids_sha256": episode_id_hash(test_ids),
            })

    configurations_frame = pd.DataFrame(configuration_rows).sort_values("configuration_id")
    episodes_frame = pd.concat(episode_frames, ignore_index=True).sort_values(
        ["configuration_id", "run_id", "eval_id"])
    folds_frame = pd.DataFrame(fold_rows).sort_values(["configuration_id", "test_run"])
    overlap_rows = []
    for vla in EXPECTED_VLAS:
        for condition in EXPECTED_CONDITIONS:
            subsets = {}
            for budget in EXPECTED_BUDGETS:
                mask = (
                    (episodes_frame["vla"] == vla)
                    & (episodes_frame["condition"] == condition)
                    & (episodes_frame["budget"] == budget)
                )
                subsets[budget] = set(episodes_frame.loc[mask, "trajectory_sha256"])
            overlap = subsets[200] & subsets[400]
            overlap_rows.append({
                "vla": vla,
                "condition": condition,
                "exact_trajectory_overlap": len(overlap),
                "unique_trajectory_hashes_200": len(subsets[200]),
                "unique_trajectory_hashes_400": len(subsets[400]),
                "semantic_scene_overlap_verifiable": False,
                "reason": "input schema has no stable scene identifier",
            })
    overlap_frame = pd.DataFrame(overlap_rows)
    configurations_frame.to_csv(audit_directory / "configurations.csv", index=False)
    episodes_frame.to_csv(
        audit_directory / "episode_manifest.csv.gz", index=False, compression="gzip")
    folds_frame.to_csv(audit_directory / "fold_manifest.csv", index=False)
    overlap_frame.to_csv(audit_directory / "budget_overlap.csv", index=False)
    report = {
        "status": "passed",
        "completed_at_utc": utc_now(),
        "elapsed_seconds": time.perf_counter() - started,
        "configurations": len(configurations_frame),
        "folds": len(folds_frame),
        "episodes": int(len(episodes_frame)),
        "failures": int(episodes_frame["failure"].sum()),
        "passes": int((episodes_frame["failure"] == 0).sum()),
        "trajectory_rows": int(configurations_frame["trajectory_rows"].sum()),
        "duplicate_episode_steps": int(
            configurations_frame["duplicate_episode_steps"].sum()),
        "exact_trajectory_overlap_200_400": int(
            overlap_frame["exact_trajectory_overlap"].sum()),
        "semantic_scene_overlap_verifiable": False,
        "phase_a_ready": True,
        "phase_b_ready": False,
        "phase_b_reason": "no causal voxel labels are present in the current input schema",
    }
    atomic_json(audit_directory / "audit.json", report)
    return report


def experiment_signature(grid: VoxelGrid, protocol_path: Path) -> Tuple[str, dict]:
    runner_path = Path(__file__).resolve()
    analysis_path = runner_path.with_name("analysis.py")
    catalogue = [
        {
            "formula_id": formula_id,
            "name": FORMULA_REGISTRY[formula_id].name,
            "expression": FORMULA_REGISTRY[formula_id].expression,
        }
        for formula_id in TABLE4_FORMULA_IDS
    ]
    details = {
        "protocol_sha256": sha256_file(protocol_path),
        "runner_sha256": sha256_file(runner_path),
        "analysis_sha256": sha256_file(analysis_path),
        "grid": grid.to_dict(),
        "formula_catalogue": catalogue,
        "formula_count": len(catalogue),
        "trajectory_aggregation": "maximum_training_voxel_empirical_percentile",
        "all_unknown_policy": "neutral_0.5",
    }
    return sha256_payload(details), details


def fold_directory(root: Path, configuration: Configuration, test_run: str) -> Path:
    return root / "folds" / configuration.configuration_id / test_run


def run_fold(
    dataset: TrajectoryDataset,
    configuration: Configuration,
    test_run: str,
    grid: VoxelGrid,
    output_root: Path,
    signature: str,
    resume: bool,
) -> dict:
    destination = fold_directory(output_root, configuration, test_run)
    completion_path = destination / "fold.json"
    if completion_path.is_file():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("experiment_signature") != signature:
            raise ValueError("completed fold has a different experiment signature: " + str(destination))
        if not resume:
            raise FileExistsError("completed fold exists and resume is disabled: " + str(destination))
        return completion

    episodes = dataset.episodes
    run_values = episodes["run_id"].astype(str)
    training = subset_dataset(dataset, episodes.loc[run_values != test_run, "episode_id"])
    testing = subset_dataset(dataset, episodes.loc[run_values == test_run, "episode_id"])
    training_ids = tuple(training.episodes["episode_id"].astype(str))
    testing_ids = tuple(testing.episodes["episode_id"].astype(str))
    if set(training_ids) & set(testing_ids):
        raise ValueError("fold contains overlapping training and test episode IDs")
    if set(training.labels.values()) != {0, 1} or set(testing.labels.values()) != {0, 1}:
        raise ValueError("fold requires both outcome classes in train and test")

    destination.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    fit_started = time.perf_counter()
    model = SBFLModel.fit(
        visits=training.visits,
        labels=training.labels,
        grid=grid,
        formula_ids=TABLE4_FORMULA_IDS,
        metadata={
            "experiment": "RQ1_phase_a_primary",
            "experiment_signature": signature,
            "configuration_id": configuration.configuration_id,
            "vla": configuration.vla,
            "condition": configuration.condition,
            "budget": configuration.budget,
            "test_run": test_run,
            "test_episode_ids_sha256": episode_id_hash(testing_ids),
        },
    )
    fit_seconds = time.perf_counter() - fit_started
    model_directory = destination / "model"
    model.save(model_directory)
    model_artifact_bytes = directory_size(model_directory)

    score_started = time.perf_counter()
    predictions = TrajectoryScorer(model).score(testing.visits, testing.episodes)
    score_seconds = time.perf_counter() - score_started
    predictions.to_csv(
        destination / "predictions.csv.gz", index=False, compression="gzip")

    metrics = evaluate_predictions(predictions, model.formula_ids)
    metrics.insert(0, "configuration_id", configuration.configuration_id)
    metrics.insert(1, "vla", configuration.vla)
    metrics.insert(2, "condition", configuration.condition)
    metrics.insert(3, "budget", configuration.budget)
    metrics.insert(4, "test_run", test_run)
    metrics["technique"] = metrics["formula_id"].map(
        {formula_id: FORMULA_REGISTRY[formula_id].name for formula_id in model.formula_ids})
    metrics["mean_known_fraction"] = float(predictions["known_fraction"].mean())
    metrics["all_unknown_trajectories"] = int(
        (predictions["scoring_status"] == "no_training_support").sum())
    metrics["score_tie_fraction"] = 1.0 - metrics["distinct_scores"] / metrics["episodes"]
    metrics["fit_seconds"] = fit_seconds
    metrics["score_seconds"] = score_seconds
    metrics["model_artifact_bytes"] = model_artifact_bytes
    metrics.to_csv(destination / "metrics.csv", index=False)

    diagnostic_rows = []
    for formula_id in model.formula_ids:
        diagnostic_rows.append({
            "configuration_id": configuration.configuration_id,
            "vla": configuration.vla,
            "condition": configuration.condition,
            "budget": configuration.budget,
            "test_run": test_run,
            "formula_id": formula_id,
            "technique": FORMULA_REGISTRY[formula_id].name,
            **model.formula_diagnostics[formula_id],
        })
    pd.DataFrame(diagnostic_rows).to_csv(destination / "formula_diagnostics.csv", index=False)
    completion = {
        "status": "complete",
        "completed_at_utc": utc_now(),
        "experiment_signature": signature,
        "configuration_id": configuration.configuration_id,
        "vla": configuration.vla,
        "condition": configuration.condition,
        "budget": configuration.budget,
        "test_run": test_run,
        "training_episodes": len(training.episodes),
        "training_failures": int(training.episodes["failure"].sum()),
        "training_passes": int((training.episodes["failure"] == 0).sum()),
        "test_episodes": len(testing.episodes),
        "test_failures": int(testing.episodes["failure"].sum()),
        "test_passes": int((testing.episodes["failure"] == 0).sum()),
        "training_episode_ids_sha256": episode_id_hash(training_ids),
        "test_episode_ids_sha256": episode_id_hash(testing_ids),
        "train_test_overlap": 0,
        "fit_seconds": fit_seconds,
        "score_seconds": score_seconds,
        "total_seconds": time.perf_counter() - started,
        "model_artifact_bytes": model_artifact_bytes,
        "mean_known_fraction": float(predictions["known_fraction"].mean()),
        "all_unknown_trajectories": int(
            (predictions["scoring_status"] == "no_training_support").sum()),
    }
    atomic_json(completion_path, completion)
    return completion


def aggregate_results(
    output_root: Path,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> dict:
    aggregate_directory = output_root / "aggregate"
    aggregate_directory.mkdir(parents=True, exist_ok=True)
    metric_paths = sorted((output_root / "folds").glob("*/*/metrics.csv"))
    diagnostic_paths = sorted(
        (output_root / "folds").glob("*/*/formula_diagnostics.csv"))
    status_paths = sorted((output_root / "folds").glob("*/*/fold.json"))
    if not metric_paths:
        raise ValueError("no completed fold metrics were found")
    fold_metrics = pd.concat([pd.read_csv(path) for path in metric_paths], ignore_index=True)
    diagnostics = pd.concat([pd.read_csv(path) for path in diagnostic_paths], ignore_index=True)
    statuses = pd.DataFrame([
        json.loads(path.read_text(encoding="utf-8")) for path in status_paths])
    key_columns = ["configuration_id", "test_run", "formula_id"]
    if fold_metrics.duplicated(key_columns).any():
        raise ValueError("aggregate input contains duplicate fold/formula rows")
    fold_metrics.to_csv(
        aggregate_directory / "fold_metrics.csv.gz", index=False, compression="gzip")
    diagnostics.to_csv(
        aggregate_directory / "formula_diagnostics.csv.gz", index=False,
        compression="gzip")
    statuses.sort_values(["configuration_id", "test_run"]).to_csv(
        aggregate_directory / "fold_status.csv", index=False)

    configuration_metrics = fold_metrics.groupby(
        ["configuration_id", "vla", "condition", "budget", "formula_id", "technique"],
        as_index=False,
    ).agg(
        roc_auc=("roc_auc", "mean"),
        average_precision=("average_precision", "mean"),
        failure_prevalence=("failure_prevalence", "mean"),
        distinct_scores=("distinct_scores", "mean"),
        mean_known_fraction=("mean_known_fraction", "mean"),
        all_unknown_trajectories=("all_unknown_trajectories", "sum"),
        score_tie_fraction=("score_tie_fraction", "mean"),
        fit_seconds=("fit_seconds", "mean"),
        score_seconds=("score_seconds", "mean"),
        model_artifact_bytes=("model_artifact_bytes", "mean"),
        folds=("test_run", "nunique"),
    )
    configuration_metrics["roc_auc_rank"] = configuration_metrics.groupby(
        "configuration_id")["roc_auc"].rank(method="average", ascending=False)
    configuration_metrics["ap_rank"] = configuration_metrics.groupby(
        "configuration_id")["average_precision"].rank(method="average", ascending=False)
    configuration_metrics["primary_block_id"] = (
        configuration_metrics["vla"].astype(str)
        + "__" + configuration_metrics["condition"].astype(str))
    configuration_metrics.to_csv(
        aggregate_directory / "configuration_metrics.csv", index=False)
    primary_block_metrics = configuration_metrics.groupby(
        ["primary_block_id", "vla", "condition", "formula_id", "technique"],
        as_index=False,
    ).agg(
        roc_auc=("roc_auc", "mean"),
        average_precision=("average_precision", "mean"),
        failure_prevalence=("failure_prevalence", "mean"),
        distinct_scores=("distinct_scores", "mean"),
        mean_known_fraction=("mean_known_fraction", "mean"),
        all_unknown_trajectories=("all_unknown_trajectories", "sum"),
        score_tie_fraction=("score_tie_fraction", "mean"),
        fit_seconds=("fit_seconds", "mean"),
        score_seconds=("score_seconds", "mean"),
        model_artifact_bytes=("model_artifact_bytes", "mean"),
        budgets=("budget", "nunique"),
        configurations=("configuration_id", "nunique"),
    )
    primary_block_metrics["roc_auc_rank"] = primary_block_metrics.groupby(
        "primary_block_id")["roc_auc"].rank(method="average", ascending=False)
    primary_block_metrics["ap_rank"] = primary_block_metrics.groupby(
        "primary_block_id")["average_precision"].rank(
            method="average", ascending=False)
    primary_block_metrics.to_csv(
        aggregate_directory / "primary_block_metrics.csv", index=False)

    overall = primary_block_metrics.groupby(
        ["formula_id", "technique"], as_index=False
    ).agg(
        macro_roc_auc=("roc_auc", "mean"),
        roc_auc_sd=("roc_auc", "std"),
        macro_average_precision=("average_precision", "mean"),
        average_precision_sd=("average_precision", "std"),
        mean_roc_auc_rank=("roc_auc_rank", "mean"),
        mean_ap_rank=("ap_rank", "mean"),
        mean_known_fraction=("mean_known_fraction", "mean"),
        mean_score_tie_fraction=("score_tie_fraction", "mean"),
        mean_fit_seconds=("fit_seconds", "mean"),
        mean_score_seconds=("score_seconds", "mean"),
        primary_blocks=("primary_block_id", "nunique"),
    ).sort_values(
        ["macro_roc_auc", "macro_average_precision", "formula_id"],
        ascending=[False, False, True], ignore_index=True)
    overall["overall_rank"] = np.arange(1, len(overall) + 1)

    subgroup_frames = []
    for dimension in ("vla", "condition", "budget"):
        grouped = configuration_metrics.groupby(
            [dimension, "formula_id", "technique"], as_index=False
        ).agg(
            macro_roc_auc=("roc_auc", "mean"),
            macro_average_precision=("average_precision", "mean"),
            mean_roc_auc_rank=("roc_auc_rank", "mean"),
            configurations=("configuration_id", "nunique"),
        )
        grouped.insert(0, "dimension", dimension)
        grouped = grouped.rename(columns={dimension: "group"})
        grouped["group"] = grouped["group"].astype(str)
        grouped["group_rank"] = grouped.groupby(
            ["dimension", "group"])["macro_roc_auc"].rank(
                method="min", ascending=False).astype(int)
        subgroup_frames.append(grouped)
    pd.concat(subgroup_frames, ignore_index=True).to_csv(
        aggregate_directory / "subgroup_metrics.csv", index=False)

    formula_ids = list(TABLE4_FORMULA_IDS)
    primary_block_order = sorted(primary_block_metrics["primary_block_id"].unique())
    primary_block_pivot = primary_block_metrics.pivot(
        index="primary_block_id", columns="formula_id", values="roc_auc"
    ).reindex(index=primary_block_order, columns=formula_ids)
    fold_metrics["primary_block_id"] = (
        fold_metrics["vla"].astype(str) + "__" + fold_metrics["condition"].astype(str))
    fold_pivot = fold_metrics.pivot(
        index=["primary_block_id", "budget", "test_run"],
        columns="formula_id", values="roc_auc"
    ).reindex(columns=formula_ids)
    fold_counts = fold_pivot.groupby(level=0).size()
    complete_design = (
        len(primary_block_order) >= 2
        and not primary_block_pivot.isna().any().any()
        and fold_counts.nunique() == 1
        and int(fold_counts.iloc[0]) >= 2
        and not fold_pivot.isna().any().any()
    )
    if not complete_design:
        overall.to_csv(aggregate_directory / "overall_ranking.csv", index=False)
        summary = {
            "status": "partial",
            "completed_folds": int(fold_metrics[["configuration_id", "test_run"]].drop_duplicates().shape[0]),
            "configurations": int(configuration_metrics["configuration_id"].nunique()),
            "primary_blocks": len(primary_block_order),
            "statistical_analysis": "not run because the paired design is incomplete",
        }
        atomic_json(aggregate_directory / "summary.json", summary)
        return summary

    friedman = friedman_test(primary_block_pivot.to_numpy(dtype=np.float64))
    leader_id = str(overall.iloc[0]["formula_id"])
    leader_index = formula_ids.index(leader_id)
    raw_p_values = {}
    wilcoxon_results = {}
    if friedman.p_value < 0.05:
        leader_scores = primary_block_pivot[leader_id].to_numpy(dtype=np.float64)
        for formula_id in formula_ids:
            if formula_id == leader_id:
                continue
            result = wilcoxon_signed_rank(
                leader_scores,
                primary_block_pivot[formula_id].to_numpy(dtype=np.float64),
            )
            raw_p_values[formula_id] = result.p_value
            wilcoxon_results[formula_id] = result
        adjusted_p_values = holm_adjust(raw_p_values)
        posthoc_performed = True
    else:
        adjusted_p_values = {formula_id: 1.0 for formula_id in formula_ids if formula_id != leader_id}
        posthoc_performed = False

    budget_order = list(EXPECTED_BUDGETS)
    run_order = list(EXPECTED_RUNS)
    fold_tensor = np.empty(
        (len(primary_block_order), len(budget_order), len(run_order), len(formula_ids)),
        dtype=np.float64,
    )
    for block_index, primary_block_id in enumerate(primary_block_order):
        for budget_index, budget in enumerate(budget_order):
            values = fold_pivot.loc[(primary_block_id, budget)].reindex(
                run_order).to_numpy(dtype=np.float64)
            if values.shape != (len(run_order), len(formula_ids)) or not np.isfinite(values).all():
                raise ValueError("incomplete budget/run bootstrap tensor")
            fold_tensor[block_index, budget_index] = values
    lower, upper, delta_lower, delta_upper = hierarchical_budget_bootstrap(
        fold_tensor,
        leader_index=leader_index,
        iterations=bootstrap_iterations,
        seed=bootstrap_seed,
    )
    bootstrap_rows = {
        formula_id: {
            "roc_auc_ci_lower": float(lower[index]),
            "roc_auc_ci_upper": float(upper[index]),
            "delta_vs_leader_ci_lower": float(delta_lower[index]),
            "delta_vs_leader_ci_upper": float(delta_upper[index]),
        }
        for index, formula_id in enumerate(formula_ids)
    }
    leader_auc = float(overall.loc[overall["formula_id"] == leader_id, "macro_roc_auc"].iloc[0])
    comparison_rows = []
    for row in overall.itertuples(index=False):
        formula_id = str(row.formula_id)
        delta = leader_auc - float(row.macro_roc_auc)
        if formula_id == leader_id:
            raw_p = 1.0
            adjusted_p = 1.0
            rank_biserial = 0.0
            nonzero_pairs = 0
        elif posthoc_performed:
            result = wilcoxon_results[formula_id]
            raw_p = result.p_value
            adjusted_p = adjusted_p_values[formula_id]
            rank_biserial = result.rank_biserial
            nonzero_pairs = result.nonzero_pairs
        else:
            raw_p = math.nan
            adjusted_p = 1.0
            rank_biserial = math.nan
            nonzero_pairs = 0
        comparison_rows.append({
            "formula_id": formula_id,
            "technique": row.technique,
            "macro_roc_auc": float(row.macro_roc_auc),
            "macro_average_precision": float(row.macro_average_precision),
            "mean_roc_auc_rank": float(row.mean_roc_auc_rank),
            "delta_auc_vs_leader": delta,
            "wilcoxon_p_value": raw_p,
            "holm_adjusted_p_value": adjusted_p,
            "rank_biserial_leader_minus_candidate": rank_biserial,
            "nonzero_primary_block_pairs": nonzero_pairs,
            "practically_equivalent": bool(delta < 0.01 and adjusted_p >= 0.05),
            **bootstrap_rows[formula_id],
        })
    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["macro_roc_auc", "formula_id"], ascending=[False, True], ignore_index=True)
    comparison["phase_a_rank"] = np.arange(1, len(comparison) + 1)
    comparison.to_csv(aggregate_directory / "statistical_comparison.csv", index=False)
    overall = overall.merge(
        comparison[[
            "formula_id", "roc_auc_ci_lower", "roc_auc_ci_upper",
            "practically_equivalent"]],
        on="formula_id", how="left", validate="one_to_one")
    overall.to_csv(aggregate_directory / "overall_ranking.csv", index=False)

    friedman_payload = {
        "test": "tie-corrected Friedman chi-square approximation",
        "statistic": friedman.statistic,
        "degrees_of_freedom": friedman.degrees_of_freedom,
        "p_value": friedman.p_value,
        "primary_blocks": friedman.blocks,
        "primary_block_definition": "VLA x trajectory condition; budgets averaged within block",
        "techniques": friedman.techniques,
        "posthoc_performed": posthoc_performed,
        "posthoc": "paired Wilcoxon signed-rank versus Phase A leader with Holm correction",
        "bootstrap": {
            "method": (
                "hierarchical: VLA-condition blocks, then equal-budget run replicates "
                "within sampled blocks"),
            "iterations": bootstrap_iterations,
            "seed": bootstrap_seed,
        },
    }
    atomic_json(aggregate_directory / "friedman_test.json", friedman_payload)
    equivalent = comparison.loc[comparison["practically_equivalent"], "formula_id"].tolist()
    summary = {
        "status": "complete",
        "completed_at_utc": utc_now(),
        "phase": "A: held-out trajectory discrimination",
        "completed_folds": int(fold_metrics[["configuration_id", "test_run"]].drop_duplicates().shape[0]),
        "configurations": int(configuration_metrics["configuration_id"].nunique()),
        "primary_blocks": len(primary_block_order),
        "budgets_treated_as_sensitivity": True,
        "techniques": len(formula_ids),
        "leader_formula_id": leader_id,
        "leader_technique": FORMULA_REGISTRY[leader_id].name,
        "leader_macro_roc_auc": leader_auc,
        "leader_macro_average_precision": float(
            overall.loc[overall["formula_id"] == leader_id, "macro_average_precision"].iloc[0]),
        "practical_equivalence_group": equivalent,
        "selection_status": (
            "Phase A leader only; final localization winner requires Phase B voxel ground truth"),
        "friedman": friedman_payload,
    }
    atomic_json(aggregate_directory / "summary.json", summary)
    top = overall.head(10)
    lines = [
        "# RQ1 Phase A results",
        "",
        "This run evaluates held-out failure discrimination from voxel suspiciousness. "
        "It does not establish causal voxel localization because Phase B ground truth is absent.",
        "",
        "| Rank | Technique | Macro ROC-AUC | 95% CI | Macro AP | Mean rank | Equivalent |",
        "|---:|---|---:|---:|---:|---:|:---:|",
    ]
    for row in top.itertuples(index=False):
        lines.append(
            "| {} | {} | {:.4f} | [{:.4f}, {:.4f}] | {:.4f} | {:.2f} | {} |".format(
                row.overall_rank,
                row.technique,
                row.macro_roc_auc,
                row.roc_auc_ci_lower,
                row.roc_auc_ci_upper,
                row.macro_average_precision,
                row.mean_roc_auc_rank,
                "yes" if row.practically_equivalent else "no",
            ))
    lines.extend([
        "",
        "Friedman: statistic={:.4f}, df={}, p={:.6g}.".format(
            friedman.statistic, friedman.degrees_of_freedom, friedman.p_value),
        "",
        "Final wording must remain **failure-associated voxel ranking** until Phase B is completed.",
        "",
    ])
    (aggregate_directory / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def git_revision() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("input"))
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("experiments/RQ1/results/phase_a_primary"))
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--origin", type=float, nargs=3, default=(-1.0, -1.0, 0.0))
    parser.add_argument("--chunk-size", type=int, default=250_000)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261023)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--configuration", action="append", default=[])
    parser.add_argument("--max-folds", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_folds is not None and args.max_folds <= 0:
        raise ValueError("--max-folds must be positive")
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    protocol_path = Path(__file__).resolve().with_name("RQ1.md")
    grid = VoxelGrid(args.voxel_size, tuple(args.origin), "robot_base")
    configurations = list(discover_configurations(args.input_root))
    if args.configuration:
        requested = set(args.configuration)
        available = {item.configuration_id for item in configurations}
        unknown = sorted(requested - available)
        if unknown:
            raise ValueError("unknown configurations: " + ", ".join(unknown))
        configurations = [
            item for item in configurations if item.configuration_id in requested]
    signature, signature_details = experiment_signature(grid, protocol_path)
    run_metadata = {
        "status": "auditing",
        "started_at_utc": utc_now(),
        "experiment_signature": signature,
        "signature_details": signature_details,
        "input_root": str(args.input_root.expanduser().resolve()),
        "output_root": str(output_root),
        "configurations": [item.configuration_id for item in configurations],
        "expected_folds": len(configurations) * (
            min(10, args.max_folds) if args.max_folds is not None else 10),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "git_revision_at_launch": git_revision(),
        "bootstrap_iterations": args.bootstrap_iterations,
        "bootstrap_seed": args.bootstrap_seed,
    }
    atomic_json(output_root / "run.json", run_metadata)
    audit_report = audit_all(
        configurations, grid, output_root / "audit", args.chunk_size)
    if args.audit_only:
        run_metadata.update({
            "status": "audit_complete",
            "completed_at_utc": utc_now(),
            "audit": audit_report,
        })
        atomic_json(output_root / "run.json", run_metadata)
        print(json.dumps(run_metadata, indent=2, sort_keys=True), flush=True)
        return 0

    run_metadata["status"] = "running"
    atomic_json(output_root / "run.json", run_metadata)
    total_folds = int(run_metadata["expected_folds"])
    completed = 0
    experiment_started = time.perf_counter()
    for configuration_index, configuration in enumerate(configurations, start=1):
        dataset = load_trajectory_dataset(
            [configuration.directory], grid=grid, require_labels=True,
            chunk_size=args.chunk_size)
        runs = list(validate_configuration_metadata(configuration, dataset.episodes))
        if args.max_folds is not None:
            runs = runs[:args.max_folds]
        for test_run in runs:
            completed += 1
            print(
                "[fold {}/{}; config {}/{}] {} test={}".format(
                    completed, total_folds, configuration_index, len(configurations),
                    configuration.configuration_id, test_run),
                flush=True,
            )
            status = run_fold(
                dataset=dataset,
                configuration=configuration,
                test_run=test_run,
                grid=grid,
                output_root=output_root,
                signature=signature,
                resume=args.resume,
            )
            print(
                "  complete: ROC inputs={} known={:.3f} time={:.2f}s".format(
                    status["test_episodes"], status["mean_known_fraction"],
                    status["total_seconds"]),
                flush=True,
            )
    print("[aggregate] computing paired statistics", flush=True)
    summary = aggregate_results(
        output_root,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
    )
    run_metadata.update({
        "status": "complete",
        "completed_at_utc": utc_now(),
        "elapsed_seconds": time.perf_counter() - experiment_started,
        "audit": audit_report,
        "summary": summary,
    })
    atomic_json(output_root / "run.json", run_metadata)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
