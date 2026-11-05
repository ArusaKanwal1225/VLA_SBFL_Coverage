#!/usr/bin/env python3
"""Run and analyze the frozen RQ2 voxel-granularity experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from experiments.RQ1.analysis import friedman_test, holm_adjust, wilcoxon_signed_rank
from experiments.RQ1.run_experiment import (
    EXPECTED_BUDGETS,
    EXPECTED_RUNS,
    sha256_file,
    validate_primary_block_metrics,
)
from experiments.RQ2.analysis import matched_size_bootstrap, spearman_rank_correlation
from vla_sbfl.formulas import FORMULA_REGISTRY, STUDY_FORMULA_IDS


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RQ1_DIRECTORY = REPOSITORY_ROOT / "experiments" / "RQ1"
RQ1_RUNNER = RQ1_DIRECTORY / "run_experiment.py"
RQ1_ANALYSIS = RQ1_DIRECTORY / "analysis.py"
RQ1_PROTOCOL = RQ1_DIRECTORY / "RQ1.md"
VOXEL_SIZES = (0.01, 0.02, 0.05)
PRIMARY_SIZE = 0.02
REFERENCE_FORMULA = "harmonic_mean"
PRACTICAL_MARGIN = 0.01


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def sha256_payload(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def size_slug(voxel_size: float) -> str:
    return "size_" + format(float(voxel_size), ".2f").replace(".", "p")


def same_size(first: float, second: float) -> bool:
    return math.isclose(float(first), float(second), rel_tol=0.0, abs_tol=1e-12)


def rq2_signature(origin: Sequence[float]) -> Tuple[str, dict]:
    protocol_path = Path(__file__).resolve().with_name("RQ2.md")
    analysis_path = Path(__file__).resolve().with_name("analysis.py")
    details = {
        "protocol_sha256": sha256_file(protocol_path),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "analysis_sha256": sha256_file(analysis_path),
        "rq1_engine": {
            "protocol_sha256": sha256_file(RQ1_PROTOCOL),
            "runner_sha256": sha256_file(RQ1_RUNNER),
            "analysis_sha256": sha256_file(RQ1_ANALYSIS),
        },
        "voxel_sizes": list(VOXEL_SIZES),
        "primary_size": PRIMARY_SIZE,
        "origin": [float(value) for value in origin],
        "reference_formula": REFERENCE_FORMULA,
        "formula_ids": list(STUDY_FORMULA_IDS),
        "practical_margin": PRACTICAL_MARGIN,
    }
    return sha256_payload(details), details


def build_cell_command(
    input_root: Path,
    output_root: Path,
    voxel_size: float,
    origin: Sequence[float],
    chunk_size: int,
    bootstrap_iterations: int,
    bootstrap_seed: int,
    resume: bool,
    smoke: bool,
) -> List[str]:
    command = [
        sys.executable,
        str(RQ1_RUNNER),
        "--input-root",
        str(input_root),
        "--output-root",
        str(output_root),
        "--voxel-size",
        str(voxel_size),
        "--origin",
        *(str(value) for value in origin),
        "--chunk-size",
        str(chunk_size),
        "--bootstrap-iterations",
        str(bootstrap_iterations),
        "--bootstrap-seed",
        str(bootstrap_seed),
        "--resume" if resume else "--no-resume",
    ]
    if smoke:
        command.extend([
            "--configuration", "EO1__DH__200",
            "--max-folds", "1",
        ])
    return command


def execute_cell(command: Sequence[str]) -> None:
    environment = os.environ.copy()
    python_paths = [str(REPOSITORY_ROOT), str(REPOSITORY_ROOT / "src_new")]
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    subprocess.run(
        list(command),
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
    )


def _validate_engine_hashes(run_metadata: Mapping[str, object], source: Path) -> None:
    signature = run_metadata.get("signature_details")
    if not isinstance(signature, Mapping):
        raise ValueError("cell has no signature details: " + str(source))
    expected = {
        "protocol_sha256": sha256_file(RQ1_PROTOCOL),
        "runner_sha256": sha256_file(RQ1_RUNNER),
        "analysis_sha256": sha256_file(RQ1_ANALYSIS),
    }
    for key, value in expected.items():
        if signature.get(key) != value:
            raise ValueError(
                "cell {} does not match the frozen RQ1 engine ({})".format(source, key))


def load_cell_fold_metrics(
    source: Path,
    voxel_size: float,
    origin: Sequence[float],
    require_full_design: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    run_path = source / "run.json"
    if not run_path.is_file():
        raise FileNotFoundError(run_path)
    run_metadata = json.loads(run_path.read_text(encoding="utf-8"))
    if run_metadata.get("status") != "complete":
        raise ValueError("cell is not complete: " + str(source))
    _validate_engine_hashes(run_metadata, source)
    signature = run_metadata["signature_details"]
    grid = signature.get("grid")
    if not isinstance(grid, Mapping):
        raise ValueError("cell has no frozen grid: " + str(source))
    if not same_size(float(grid.get("voxel_size", math.nan)), voxel_size):
        raise ValueError("cell voxel size does not match its RQ2 factor: " + str(source))
    actual_origin = tuple(float(value) for value in grid.get("origin", ()))
    expected_origin = tuple(float(value) for value in origin)
    if actual_origin != expected_origin or grid.get("coordinate_frame") != "robot_base":
        raise ValueError("cell grid origin/frame does not match RQ2: " + str(source))

    metrics_path = source / "aggregate" / "fold_metrics.csv.gz"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    frame = pd.read_csv(metrics_path)
    required = {
        "configuration_id", "vla", "condition", "budget", "test_run",
        "formula_id", "technique", "roc_auc", "average_precision", "episodes",
        "failures", "failure_prevalence", "mean_known_fraction",
        "all_unknown_trajectories", "score_tie_fraction", "fit_seconds",
        "score_seconds", "model_artifact_bytes",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            "cell metrics missing columns {}: {}".format(sorted(missing), source))
    keys = ["configuration_id", "test_run", "formula_id"]
    if frame.duplicated(keys).any():
        raise ValueError("cell contains duplicate fold/formula rows: " + str(source))
    if set(frame["formula_id"].astype(str)) != set(STUDY_FORMULA_IDS):
        raise ValueError("cell formula catalogue differs from RQ1: " + str(source))
    numeric = frame[[
        "roc_auc", "average_precision", "failure_prevalence",
        "mean_known_fraction", "score_tie_fraction", "fit_seconds",
        "score_seconds", "model_artifact_bytes",
    ]].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("cell metrics contain non-finite values: " + str(source))
    folds = frame[["configuration_id", "test_run"]].drop_duplicates()
    if require_full_design:
        if len(folds) != 320 or frame["configuration_id"].nunique() != 32:
            raise ValueError("cell does not contain the frozen 320-fold design: " + str(source))
        counts = folds.groupby("configuration_id")["test_run"].nunique()
        if (
            set(counts.index) != set(frame["configuration_id"].unique())
            or set(counts.to_list()) != {10}
        ):
            raise ValueError("cell does not contain ten folds per configuration: " + str(source))
        if len(frame) != 320 * len(STUDY_FORMULA_IDS):
            raise ValueError("cell does not contain 34 metrics per fold: " + str(source))
    frame.insert(0, "voxel_size", float(voxel_size))
    return frame, run_metadata


def validate_matched_fold_identities(frames: Mapping[float, pd.DataFrame]) -> None:
    identity_columns = [
        "configuration_id", "vla", "condition", "budget", "test_run",
        "formula_id", "technique", "episodes", "failures", "failure_prevalence",
    ]
    primary = frames[PRIMARY_SIZE][identity_columns].sort_values(
        ["configuration_id", "test_run", "formula_id"]
    ).reset_index(drop=True)
    for voxel_size, frame in frames.items():
        candidate = frame[identity_columns].sort_values(
            ["configuration_id", "test_run", "formula_id"]
        ).reset_index(drop=True)
        if not candidate.equals(primary):
            raise ValueError(
                "fold identities or outcomes differ at voxel size {}".format(voxel_size))


def load_training_voxel_counts(source: Path, voxel_size: float) -> pd.DataFrame:
    rows = []
    for metadata_path in sorted((source / "folds").glob("*/*/model/metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        grid = metadata.get("grid", {})
        training = metadata.get("training", {})
        if not same_size(float(grid.get("voxel_size", math.nan)), voxel_size):
            raise ValueError("model metadata has an unexpected voxel size: " + str(metadata_path))
        rows.append({
            "voxel_size": float(voxel_size),
            "configuration_id": str(training["configuration_id"]),
            "test_run": str(training["test_run"]),
            "training_voxels": int(training["training_voxels"]),
        })
    frame = pd.DataFrame(rows)
    if len(frame) != 320 or frame.duplicated(
        ["configuration_id", "test_run"]).any():
        raise ValueError("model metadata does not cover 320 unique folds: " + str(source))
    return frame


def _build_fold_tensor(
    fold_metrics: pd.DataFrame,
    sizes: Sequence[float],
    formula_ids: Sequence[str],
) -> Tuple[np.ndarray, List[str]]:
    frame = fold_metrics.copy()
    frame["primary_block_id"] = (
        frame["vla"].astype(str) + "__" + frame["condition"].astype(str))
    block_order = sorted(frame["primary_block_id"].unique())
    columns = pd.MultiIndex.from_product(
        [list(sizes), list(formula_ids)], names=["voxel_size", "formula_id"])
    pivot = frame.pivot(
        index=["primary_block_id", "budget", "test_run"],
        columns=["voxel_size", "formula_id"],
        values="roc_auc",
    ).reindex(columns=columns)
    full_index = pd.MultiIndex.from_product(
        [block_order, list(EXPECTED_BUDGETS), list(EXPECTED_RUNS)],
        names=["primary_block_id", "budget", "test_run"],
    )
    pivot = pivot.reindex(full_index)
    if pivot.isna().any().any():
        raise ValueError("cross-size fold tensor is incomplete")
    values = pivot.to_numpy(dtype=np.float64).reshape(
        len(block_order), len(EXPECTED_BUDGETS), len(EXPECTED_RUNS),
        len(sizes), len(formula_ids))
    return values, block_order


def _primary_block_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    configuration = fold_metrics.groupby(
        ["voxel_size", "configuration_id", "vla", "condition", "budget",
         "formula_id", "technique"],
        as_index=False,
    ).agg(
        roc_auc=("roc_auc", "mean"),
        average_precision=("average_precision", "mean"),
        failure_prevalence=("failure_prevalence", "mean"),
        mean_known_fraction=("mean_known_fraction", "mean"),
        score_tie_fraction=("score_tie_fraction", "mean"),
        folds=("test_run", "nunique"),
    )
    configuration["primary_block_id"] = (
        configuration["vla"].astype(str)
        + "__" + configuration["condition"].astype(str))
    return configuration.groupby(
        ["voxel_size", "primary_block_id", "vla", "condition",
         "formula_id", "technique"],
        as_index=False,
    ).agg(
        roc_auc=("roc_auc", "mean"),
        average_precision=("average_precision", "mean"),
        failure_prevalence=("failure_prevalence", "mean"),
        mean_known_fraction=("mean_known_fraction", "mean"),
        score_tie_fraction=("score_tie_fraction", "mean"),
        budgets=("budget", "nunique"),
    )


def _primary_tensor(
    primary_metrics: pd.DataFrame,
    block_order: Sequence[str],
    sizes: Sequence[float],
    formula_ids: Sequence[str],
) -> np.ndarray:
    columns = pd.MultiIndex.from_product(
        [list(sizes), list(formula_ids)], names=["voxel_size", "formula_id"])
    pivot = primary_metrics.pivot(
        index="primary_block_id",
        columns=["voxel_size", "formula_id"],
        values="roc_auc",
    ).reindex(index=list(block_order), columns=columns)
    if pivot.isna().any().any():
        raise ValueError("cross-size primary-block tensor is incomplete")
    return pivot.to_numpy(dtype=np.float64).reshape(
        len(block_order), len(sizes), len(formula_ids))


def aggregate_cross_size(
    cell_sources: Mapping[float, Path],
    output_root: Path,
    origin: Sequence[float],
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> dict:
    aggregate_directory = output_root / "aggregate"
    aggregate_directory.mkdir(parents=True, exist_ok=True)
    frames: Dict[float, pd.DataFrame] = {}
    cell_metadata: Dict[str, dict] = {}
    voxel_counts = []
    for voxel_size in VOXEL_SIZES:
        frame, metadata = load_cell_fold_metrics(
            cell_sources[voxel_size], voxel_size, origin, require_full_design=True)
        frames[voxel_size] = frame
        cell_metadata[str(voxel_size)] = {
            "source": str(cell_sources[voxel_size].resolve()),
            "experiment_signature": metadata["experiment_signature"],
            "run_status": metadata["status"],
        }
        voxel_counts.append(load_training_voxel_counts(
            cell_sources[voxel_size], voxel_size))
    validate_matched_fold_identities(frames)
    fold_metrics = pd.concat(
        [frames[size] for size in VOXEL_SIZES], ignore_index=True)
    fold_metrics.to_csv(
        aggregate_directory / "fold_metrics_by_size.csv.gz",
        index=False,
        compression="gzip",
    )

    primary_metrics = _primary_block_metrics(fold_metrics)
    primary_metrics.to_csv(
        aggregate_directory / "primary_block_metrics.csv", index=False)
    formula_ids = list(STUDY_FORMULA_IDS)
    sizes = list(VOXEL_SIZES)
    reference_index = formula_ids.index(REFERENCE_FORMULA)
    primary_size_index = sizes.index(PRIMARY_SIZE)
    fold_tensor, block_order = _build_fold_tensor(fold_metrics, sizes, formula_ids)
    block_tensor = _primary_tensor(primary_metrics, block_order, sizes, formula_ids)
    bootstrap = matched_size_bootstrap(
        fold_tensor,
        primary_size_index=primary_size_index,
        reference_technique_index=reference_index,
        iterations=bootstrap_iterations,
        seed=bootstrap_seed,
    )

    overall = primary_metrics.groupby(
        ["voxel_size", "formula_id", "technique"], as_index=False
    ).agg(
        macro_roc_auc=("roc_auc", "mean"),
        macro_average_precision=("average_precision", "mean"),
        mean_known_fraction=("mean_known_fraction", "mean"),
        mean_score_tie_fraction=("score_tie_fraction", "mean"),
        primary_blocks=("primary_block_id", "nunique"),
    )
    overall["within_size_rank"] = overall.groupby("voxel_size")[
        "macro_roc_auc"].rank(method="min", ascending=False).astype(int)
    ci_rows = []
    for size_index, voxel_size in enumerate(sizes):
        for formula_index, formula_id in enumerate(formula_ids):
            ci_rows.append({
                "voxel_size": voxel_size,
                "formula_id": formula_id,
                "roc_auc_ci_lower": bootstrap.score_lower[size_index, formula_index],
                "roc_auc_ci_upper": bootstrap.score_upper[size_index, formula_index],
            })
    overall = overall.merge(
        pd.DataFrame(ci_rows),
        on=["voxel_size", "formula_id"],
        validate="one_to_one",
    ).sort_values(
        ["voxel_size", "within_size_rank", "formula_id"], ignore_index=True)
    overall.to_csv(aggregate_directory / "rankings_by_size.csv", index=False)

    reference_scores = block_tensor[:, :, reference_index]
    size_friedman = friedman_test(reference_scores)
    reference_means = reference_scores.mean(axis=0)
    leader_size_index = int(np.argmax(reference_means))
    leader_size = sizes[leader_size_index]
    pair_results = {}
    raw_size_p_values = {}
    if size_friedman.p_value < 0.05:
        for size_index, voxel_size in enumerate(sizes):
            if size_index == leader_size_index:
                continue
            result = wilcoxon_signed_rank(
                reference_scores[:, leader_size_index],
                reference_scores[:, size_index],
            )
            pair_results[voxel_size] = result
            raw_size_p_values[str(voxel_size)] = result.p_value
        adjusted_size_p_values = holm_adjust(raw_size_p_values)
        size_posthoc = True
    else:
        adjusted_size_p_values = {
            str(size): 1.0 for size in sizes if not same_size(size, leader_size)}
        size_posthoc = False

    size_rows = []
    for size_index, voxel_size in enumerate(sizes):
        delta_vs_primary = float(
            reference_means[size_index] - reference_means[primary_size_index])
        delta_vs_leader = float(
            reference_means[leader_size_index] - reference_means[size_index])
        if size_index == leader_size_index:
            raw_p = 1.0
            adjusted_p = 1.0
            rank_biserial = 0.0
        elif size_posthoc:
            result = pair_results[voxel_size]
            raw_p = result.p_value
            adjusted_p = adjusted_size_p_values[str(voxel_size)]
            rank_biserial = result.rank_biserial
        else:
            raw_p = math.nan
            adjusted_p = 1.0
            rank_biserial = math.nan
        size_rows.append({
            "voxel_size": voxel_size,
            "macro_roc_auc": float(reference_means[size_index]),
            "roc_auc_ci_lower": bootstrap.score_lower[size_index, reference_index],
            "roc_auc_ci_upper": bootstrap.score_upper[size_index, reference_index],
            "delta_auc_vs_primary_0p02": delta_vs_primary,
            "delta_vs_primary_ci_lower": bootstrap.delta_lower[
                size_index, reference_index],
            "delta_vs_primary_ci_upper": bootstrap.delta_upper[
                size_index, reference_index],
            "delta_auc_vs_size_leader": delta_vs_leader,
            "wilcoxon_p_value_vs_size_leader": raw_p,
            "holm_adjusted_p_value": adjusted_p,
            "rank_biserial_size_leader_minus_candidate": rank_biserial,
            "practically_equivalent_to_size_leader": bool(
                delta_vs_leader < PRACTICAL_MARGIN and adjusted_p >= 0.05),
        })
    size_comparison = pd.DataFrame(size_rows).sort_values(
        ["macro_roc_auc", "voxel_size"], ascending=[False, True], ignore_index=True)
    size_comparison["phase_a_size_rank"] = np.arange(1, len(size_comparison) + 1)
    size_comparison.to_csv(
        aggregate_directory / "voxel_size_comparison.csv", index=False)

    size_change_results = {}
    size_change_rows = []
    for size_index, voxel_size in enumerate(sizes):
        if size_index == primary_size_index:
            continue
        for formula_index, formula_id in enumerate(formula_ids):
            key = "{}|{}".format(voxel_size, formula_id)
            result = wilcoxon_signed_rank(
                block_tensor[:, size_index, formula_index],
                block_tensor[:, primary_size_index, formula_index],
            )
            size_change_results[key] = result
    adjusted_size_changes = holm_adjust({
        key: result.p_value for key, result in size_change_results.items()})
    for size_index, voxel_size in enumerate(sizes):
        if size_index == primary_size_index:
            continue
        for formula_index, formula_id in enumerate(formula_ids):
            key = "{}|{}".format(voxel_size, formula_id)
            result = size_change_results[key]
            delta = float(np.mean(
                block_tensor[:, size_index, formula_index]
                - block_tensor[:, primary_size_index, formula_index]))
            adjusted_p = adjusted_size_changes[key]
            size_change_rows.append({
                "voxel_size": voxel_size,
                "reference_voxel_size": PRIMARY_SIZE,
                "formula_id": formula_id,
                "technique": FORMULA_REGISTRY[formula_id].name,
                "delta_macro_roc_auc": delta,
                "delta_ci_lower": bootstrap.delta_lower[size_index, formula_index],
                "delta_ci_upper": bootstrap.delta_upper[size_index, formula_index],
                "wilcoxon_p_value": result.p_value,
                "holm_adjusted_p_value": adjusted_p,
                "rank_biserial_size_minus_primary": result.rank_biserial,
                "practically_meaningful_change": bool(
                    abs(delta) >= PRACTICAL_MARGIN and adjusted_p < 0.05),
            })
    technique_size_effects = pd.DataFrame(size_change_rows).sort_values(
        ["voxel_size", "delta_macro_roc_auc", "formula_id"],
        ascending=[True, False, True],
        ignore_index=True,
    )
    technique_size_effects.to_csv(
        aggregate_directory / "technique_size_effects.csv", index=False)

    interaction_results = {}
    interaction_rows = []
    for size_index, voxel_size in enumerate(sizes):
        if size_index == primary_size_index:
            continue
        reference_delta = (
            block_tensor[:, size_index, reference_index]
            - block_tensor[:, primary_size_index, reference_index])
        for formula_index, formula_id in enumerate(formula_ids):
            if formula_index == reference_index:
                continue
            technique_delta = (
                block_tensor[:, size_index, formula_index]
                - block_tensor[:, primary_size_index, formula_index])
            contrast = technique_delta - reference_delta
            key = "{}|{}".format(voxel_size, formula_id)
            interaction_results[key] = (
                wilcoxon_signed_rank(contrast, np.zeros_like(contrast)),
                float(contrast.mean()),
                size_index,
                formula_index,
            )
    adjusted_interactions = holm_adjust({
        key: value[0].p_value for key, value in interaction_results.items()})
    for key, (result, interaction, size_index, formula_index) in interaction_results.items():
        voxel_size = sizes[size_index]
        formula_id = formula_ids[formula_index]
        adjusted_p = adjusted_interactions[key]
        interaction_rows.append({
            "voxel_size": voxel_size,
            "reference_voxel_size": PRIMARY_SIZE,
            "formula_id": formula_id,
            "technique": FORMULA_REGISTRY[formula_id].name,
            "reference_technique": FORMULA_REGISTRY[REFERENCE_FORMULA].name,
            "difference_in_differences_roc_auc": interaction,
            "interaction_ci_lower": bootstrap.interaction_lower[
                size_index, formula_index],
            "interaction_ci_upper": bootstrap.interaction_upper[
                size_index, formula_index],
            "wilcoxon_p_value": result.p_value,
            "holm_adjusted_p_value": adjusted_p,
            "rank_biserial_interaction": result.rank_biserial,
            "practically_meaningful_interaction": bool(
                abs(interaction) >= PRACTICAL_MARGIN and adjusted_p < 0.05),
        })
    interactions = pd.DataFrame(interaction_rows).sort_values(
        ["voxel_size", "difference_in_differences_roc_auc", "formula_id"],
        ascending=[True, False, True],
        ignore_index=True,
    )
    interactions.to_csv(
        aggregate_directory / "technique_size_interactions.csv", index=False)

    rank_rows = []
    overall_lookup = overall.pivot(
        index="formula_id", columns="voxel_size", values="macro_roc_auc"
    ).reindex(index=formula_ids, columns=sizes)
    for first, second in combinations(sizes, 2):
        rank_rows.append({
            "first_voxel_size": first,
            "second_voxel_size": second,
            "spearman_rank_correlation": spearman_rank_correlation(
                overall_lookup[first], overall_lookup[second]),
            "techniques": len(formula_ids),
        })
    rank_stability = pd.DataFrame(rank_rows)
    rank_stability.to_csv(
        aggregate_directory / "ranking_stability.csv", index=False)

    voxel_count_frame = pd.concat(voxel_counts, ignore_index=True)
    reference_folds = fold_metrics[
        fold_metrics["formula_id"] == REFERENCE_FORMULA].copy()
    diagnostics = reference_folds.groupby("voxel_size", as_index=False).agg(
        mean_known_fraction=("mean_known_fraction", "mean"),
        all_unknown_trajectories=("all_unknown_trajectories", "sum"),
        mean_score_tie_fraction=("score_tie_fraction", "mean"),
        mean_fit_seconds=("fit_seconds", "mean"),
        mean_score_seconds=("score_seconds", "mean"),
        mean_model_artifact_bytes=("model_artifact_bytes", "mean"),
        folds=("test_run", "size"),
    ).merge(
        voxel_count_frame.groupby("voxel_size", as_index=False).agg(
            mean_training_voxels=("training_voxels", "mean"),
            median_training_voxels=("training_voxels", "median"),
            minimum_training_voxels=("training_voxels", "min"),
            maximum_training_voxels=("training_voxels", "max"),
        ),
        on="voxel_size",
        validate="one_to_one",
    )
    diagnostics.to_csv(aggregate_directory / "size_diagnostics.csv", index=False)

    meaningful_interactions = interactions[
        interactions["practically_meaningful_interaction"]]
    winners = {}
    for voxel_size in sizes:
        cell = overall[overall["voxel_size"] == voxel_size].sort_values(
            ["macro_roc_auc", "macro_average_precision", "formula_id"],
            ascending=[False, False, True],
        )
        winners[str(voxel_size)] = {
            "formula_id": str(cell.iloc[0]["formula_id"]),
            "technique": str(cell.iloc[0]["technique"]),
            "macro_roc_auc": float(cell.iloc[0]["macro_roc_auc"]),
        }
    size_equivalence_group = size_comparison.loc[
        size_comparison["practically_equivalent_to_size_leader"],
        "voxel_size",
    ].astype(float).tolist()
    tests = {
        "size_friedman": {
            "test": "tie-corrected Friedman chi-square approximation",
            "reference_formula": REFERENCE_FORMULA,
            "statistic": size_friedman.statistic,
            "degrees_of_freedom": size_friedman.degrees_of_freedom,
            "p_value": size_friedman.p_value,
            "primary_blocks": size_friedman.blocks,
            "posthoc_performed": size_posthoc,
        },
        "size_change_holm_family": len(size_change_results),
        "interaction_holm_family": len(interaction_results),
        "bootstrap": {
            "method": (
                "matched hierarchical: VLA-condition blocks, then run folds "
                "within equal-weight budgets; identical draws across sizes/techniques"),
            "iterations": bootstrap_iterations,
            "seed": bootstrap_seed,
        },
    }
    atomic_json(aggregate_directory / "statistical_tests.json", tests)
    summary = {
        "status": "complete",
        "completed_at_utc": utc_now(),
        "phase": "A: held-out trajectory discrimination",
        "voxel_sizes": sizes,
        "primary_size": PRIMARY_SIZE,
        "reference_formula": REFERENCE_FORMULA,
        "primary_blocks": len(block_order),
        "folds_per_size": 320,
        "techniques": len(formula_ids),
        "phase_a_size_leader": leader_size,
        "phase_a_size_leader_macro_roc_auc": float(reference_means[leader_size_index]),
        "size_practical_equivalence_group": size_equivalence_group,
        "winner_by_size": winners,
        "meaningful_interactions": int(len(meaningful_interactions)),
        "localization_selection_status": (
            "pending Phase B causal/problematic-region ground truth"),
        "cell_sources": cell_metadata,
        "statistical_tests": tests,
    }
    atomic_json(aggregate_directory / "summary.json", summary)

    lines = [
        "# RQ2 Phase A results",
        "",
        "This analysis compares voxel sizes for held-out failure discrimination. "
        "It does not determine spatial localization quality without Phase B ground truth.",
        "",
        "## Voxel-size comparison for Harmonic Mean",
        "",
        "| Rank | Voxel size | Macro ROC-AUC | 95% CI | Delta vs 0.02 | Equivalent |",
        "|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in size_comparison.itertuples(index=False):
        lines.append(
            "| {} | {:.2f} | {:.4f} | [{:.4f}, {:.4f}] | {:+.4f} | {} |".format(
                row.phase_a_size_rank,
                row.voxel_size,
                row.macro_roc_auc,
                row.roc_auc_ci_lower,
                row.roc_auc_ci_upper,
                row.delta_auc_vs_primary_0p02,
                "yes" if row.practically_equivalent_to_size_leader else "no",
            ))
    lines.extend([
        "",
        "Friedman: statistic={:.4f}, df={}, p={:.6g}.".format(
            size_friedman.statistic,
            size_friedman.degrees_of_freedom,
            size_friedman.p_value,
        ),
        "",
        "## Technique leader by voxel size",
        "",
        "| Voxel size | Technique | Macro ROC-AUC |",
        "|---:|---|---:|",
    ])
    for voxel_size in sizes:
        winner = winners[str(voxel_size)]
        lines.append("| {:.2f} | {} | {:.4f} |".format(
            voxel_size, winner["technique"], winner["macro_roc_auc"]))
    lines.extend([
        "",
        "Practically meaningful technique-by-size interactions: **{}**.".format(
            len(meaningful_interactions)),
        "",
        "Final wording must remain **failure-associated voxel ranking** until "
        "Phase B is completed.",
        "",
    ])
    (aggregate_directory / "RESULTS.md").write_text(
        "\n".join(lines), encoding="utf-8")
    return summary


def regenerate_statistical_review(
    aggregate_directory: Path,
    method: str = "exact",
) -> dict:
    """Recompute paired tests across the frozen 4-VLA x 4-strategy design.

    Holm covers 68 size effects and 66 interactions separately. Retained
    bootstrap intervals are pointwise, not simultaneous or selection-adjusted.
    Practical-meaningfulness labels combine observed mean contrasts with the
    corrected signed-rank decision; they are not formal equivalence tests.
    """

    if method not in {"approximate", "exact"}:
        raise ValueError("Wilcoxon method must be 'approximate' or 'exact'")
    aggregate_directory = aggregate_directory.expanduser().resolve()
    primary_path = aggregate_directory / "primary_block_metrics.csv"
    effects_history_path = aggregate_directory / "technique_size_effects.csv"
    interactions_history_path = aggregate_directory / "technique_size_interactions.csv"
    for path in (primary_path, effects_history_path, interactions_history_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    primary = pd.read_csv(primary_path)
    effects_history = pd.read_csv(effects_history_path)
    interactions_history = pd.read_csv(interactions_history_path)
    required = {"voxel_size", "primary_block_id", "formula_id", "technique", "roc_auc"}
    if required - set(primary.columns):
        raise ValueError("voxel-size primary block metrics have an incompatible schema")
    sizes = sorted(primary["voxel_size"].astype(float).unique())
    formula_ids = sorted(primary["formula_id"].astype(str).unique())
    if tuple(sizes) != VOXEL_SIZES:
        raise ValueError("the frozen voxel-size design requires exactly 0.01, 0.02 and 0.05")
    for size in sizes:
        validate_primary_block_metrics(primary.loc[primary.voxel_size == size])
    if len(formula_ids) != 34 or primary["primary_block_id"].nunique() != 16:
        raise ValueError("the frozen voxel-size review requires 34 techniques and 16 blocks")
    pivot = primary.pivot(
        index="primary_block_id", columns=["voxel_size", "formula_id"], values="roc_auc")
    expected_columns = pd.MultiIndex.from_product([sizes, formula_ids])
    pivot = pivot.reindex(columns=expected_columns)
    if pivot.shape != (16, 102) or pivot.isna().any().any():
        raise ValueError("the frozen voxel-size paired design is incomplete")

    nonprimary_sizes = [size for size in sizes if not same_size(size, PRIMARY_SIZE)]
    effect_results = {}
    for size in nonprimary_sizes:
        for formula_id in formula_ids:
            key = (float(size), formula_id)
            effect_results[key] = wilcoxon_signed_rank(
                pivot[(size, formula_id)].to_numpy(dtype=np.float64),
                pivot[(PRIMARY_SIZE, formula_id)].to_numpy(dtype=np.float64),
                method=method,
            )
    if len(effect_results) != 68:
        raise AssertionError("the technique-size family must contain 68 comparisons")
    effect_adjusted = holm_adjust({str(key): value.p_value for key, value in effect_results.items()})
    effect_rows = []
    effect_history = effects_history.set_index(["voxel_size", "formula_id"], drop=False)
    for key, result in effect_results.items():
        if key not in effect_history.index:
            raise ValueError("historical technique-size results do not match the frozen design")
        row = effect_history.loc[key].to_dict()
        observed_delta = float((pivot[key] - pivot[(PRIMARY_SIZE, key[1])]).mean())
        if not math.isclose(observed_delta, float(row["delta_macro_roc_auc"]), rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("historical technique-size ROC-AUC deltas do not reconcile")
        adjusted_p = effect_adjusted[str(key)]
        row.update({
            "wilcoxon_p_value": result.p_value,
            "holm_adjusted_p_value": adjusted_p,
            "rank_biserial_size_minus_primary": result.rank_biserial,
            "practically_meaningful_change": bool(
                abs(observed_delta) >= PRACTICAL_MARGIN and adjusted_p < 0.05),
            "nonzero_pairs": result.nonzero_pairs,
            "wilcoxon_method": method,
            "holm_family_size": 68,
            "significant_0p05": adjusted_p < 0.05,
        })
        effect_rows.append(row)
    revised_effects = pd.DataFrame(effect_rows).sort_values(
        ["voxel_size", "delta_macro_roc_auc", "formula_id"],
        ascending=[True, False, True], ignore_index=True)

    interaction_results = {}
    for size in nonprimary_sizes:
        reference_delta = (
            pivot[(size, REFERENCE_FORMULA)] - pivot[(PRIMARY_SIZE, REFERENCE_FORMULA)])
        for formula_id in formula_ids:
            if formula_id == REFERENCE_FORMULA:
                continue
            technique_delta = pivot[(size, formula_id)] - pivot[(PRIMARY_SIZE, formula_id)]
            contrast = (technique_delta - reference_delta).to_numpy(dtype=np.float64)
            key = (float(size), formula_id)
            interaction_results[key] = wilcoxon_signed_rank(
                contrast, np.zeros_like(contrast), method=method)
    if len(interaction_results) != 66:
        raise AssertionError("the technique-size interaction family must contain 66 comparisons")
    interaction_adjusted = holm_adjust({
        str(key): value.p_value for key, value in interaction_results.items()})
    interaction_rows = []
    interaction_history = interactions_history.set_index(
        ["voxel_size", "formula_id"], drop=False)
    for key, result in interaction_results.items():
        if key not in interaction_history.index:
            raise ValueError("historical size interactions do not match the frozen design")
        row = interaction_history.loc[key].to_dict()
        reference_delta = (
            pivot[(key[0], REFERENCE_FORMULA)] - pivot[(PRIMARY_SIZE, REFERENCE_FORMULA)])
        technique_delta = pivot[key] - pivot[(PRIMARY_SIZE, key[1])]
        observed_interaction = float((technique_delta - reference_delta).mean())
        if not math.isclose(
            observed_interaction, float(row["difference_in_differences_roc_auc"]),
            rel_tol=0.0, abs_tol=1e-15,
        ):
            raise ValueError("historical technique-size interactions do not reconcile")
        adjusted_p = interaction_adjusted[str(key)]
        row.update({
            "wilcoxon_p_value": result.p_value,
            "holm_adjusted_p_value": adjusted_p,
            "rank_biserial_interaction": result.rank_biserial,
            "practically_meaningful_interaction": bool(
                abs(observed_interaction) >= PRACTICAL_MARGIN and adjusted_p < 0.05),
            "nonzero_pairs": result.nonzero_pairs,
            "wilcoxon_method": method,
            "holm_family_size": 66,
            "significant_0p05": adjusted_p < 0.05,
        })
        interaction_rows.append(row)
    revised_interactions = pd.DataFrame(interaction_rows).sort_values(
        ["voxel_size", "difference_in_differences_roc_auc", "formula_id"],
        ascending=[True, False, True], ignore_index=True)

    effects_output = aggregate_directory / ("technique_size_effects_" + method + ".csv")
    interactions_output = aggregate_directory / (
        "technique_size_interactions_" + method + ".csv")
    revised_effects.to_csv(effects_output, index=False)
    revised_interactions.to_csv(interactions_output, index=False)
    payload = {
        "status": "complete",
        "wilcoxon_method": method,
        "wilcoxon_alternative": "two-sided",
        "primary_blocks": 16,
        "families": {
            "technique_size_effects": {
                "comparison_count": 68,
                "holm_family_size": 68,
                "significant_comparisons_0p05": int(revised_effects["significant_0p05"].sum()),
                "output": effects_output.name,
            },
            "technique_size_interactions": {
                "comparison_count": 66,
                "holm_family_size": 66,
                "significant_comparisons_0p05": int(
                    revised_interactions["significant_0p05"].sum()),
                "output": interactions_output.name,
            },
        },
        "friedman_test": "unchanged; retained in the historical statistical_tests.json",
        "source_sha256": {
            path.name: sha256_file(path)
            for path in (primary_path, effects_history_path, interactions_history_path)
        },
        "output_sha256": {
            effects_output.name: sha256_file(effects_output),
            interactions_output.name: sha256_file(interactions_output),
        },
        "z_score_reported": False,
    }
    atomic_json(aggregate_directory / ("statistical_review_" + method + ".json"), payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("input"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/RQ2/results/phase_a_granularity"),
    )
    parser.add_argument(
        "--primary-results",
        type=Path,
        default=Path("experiments/RQ1/results/phase_a_primary"),
    )
    parser.add_argument("--origin", type=float, nargs=3, default=(-1.0, -1.0, 0.0))
    parser.add_argument("--chunk-size", type=int, default=250_000)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261023)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument(
        "--statistical-review-only",
        action="store_true",
        help="Recompute only method-labelled tests from frozen aggregate block metrics.",
    )
    parser.add_argument(
        "--wilcoxon-method", choices=("approximate", "exact"), default="exact")
    parser.add_argument(
        "--rerun-primary",
        action="store_true",
        help="Recompute 0.02 inside RQ2 instead of reusing the complete RQ1 cell.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one EO1/DH/200 fold at 0.01 and 0.05 without cross-size inference.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.bootstrap_iterations < 100:
        raise ValueError("--bootstrap-iterations must be at least 100")
    if args.statistical_review_only:
        if args.analysis_only or args.smoke or args.rerun_primary:
            raise ValueError("statistical review cannot be combined with experiment modes")
        output_root = args.output_root.expanduser().resolve()
        if not output_root.is_dir():
            raise FileNotFoundError(output_root)
        summary = regenerate_statistical_review(
            output_root / "aggregate", method=args.wilcoxon_method)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return 0
    if args.analysis_only and args.smoke:
        raise ValueError("--analysis-only and --smoke cannot be combined")
    output_root = args.output_root.expanduser().resolve()
    if args.smoke:
        output_root = output_root.parent / "smoke"
    output_root.mkdir(parents=True, exist_ok=True)
    input_root = args.input_root.expanduser().resolve()
    primary_results = args.primary_results.expanduser().resolve()
    signature, signature_details = rq2_signature(args.origin)
    metadata = {
        "status": "prepared",
        "started_at_utc": utc_now(),
        "experiment_signature": signature,
        "signature_details": signature_details,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "bootstrap_iterations": args.bootstrap_iterations,
        "bootstrap_seed": args.bootstrap_seed,
        "resume": args.resume,
        "smoke": args.smoke,
    }
    atomic_json(output_root / "run.json", metadata)
    started = time.perf_counter()

    cell_sources: Dict[float, Path] = {}
    commands = []
    sizes_to_run = (0.01, 0.05) if not args.rerun_primary else VOXEL_SIZES
    for voxel_size in sizes_to_run:
        destination = output_root / "cells" / size_slug(voxel_size)
        cell_sources[voxel_size] = destination
        command = build_cell_command(
            input_root=input_root,
            output_root=destination,
            voxel_size=voxel_size,
            origin=args.origin,
            chunk_size=args.chunk_size,
            bootstrap_iterations=args.bootstrap_iterations,
            bootstrap_seed=args.bootstrap_seed,
            resume=args.resume,
            smoke=args.smoke,
        )
        commands.append(command)
        if not args.analysis_only:
            print("[RQ2 cell] voxel_size={:.2f}".format(voxel_size), flush=True)
            execute_cell(command)
    if not args.rerun_primary and not args.smoke:
        cell_sources[PRIMARY_SIZE] = primary_results

    metadata["cell_commands"] = commands
    if args.smoke:
        for voxel_size, source in cell_sources.items():
            load_cell_fold_metrics(
                source, voxel_size, args.origin, require_full_design=False)
        metadata.update({
            "status": "smoke_complete",
            "completed_at_utc": utc_now(),
            "elapsed_seconds": time.perf_counter() - started,
            "smoke_cells": {str(size): str(path) for size, path in cell_sources.items()},
        })
        atomic_json(output_root / "run.json", metadata)
        print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)
        return 0

    missing_sizes = [size for size in VOXEL_SIZES if size not in cell_sources]
    if missing_sizes:
        raise ValueError("missing RQ2 cells: " + ", ".join(map(str, missing_sizes)))
    metadata["status"] = "analyzing"
    atomic_json(output_root / "run.json", metadata)
    summary = aggregate_cross_size(
        cell_sources=cell_sources,
        output_root=output_root,
        origin=args.origin,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
    )
    metadata.update({
        "status": "complete",
        "completed_at_utc": utc_now(),
        "elapsed_seconds": time.perf_counter() - started,
        "cell_sources": {str(size): str(path) for size, path in cell_sources.items()},
        "summary": summary,
    })
    atomic_json(output_root / "run.json", metadata)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
