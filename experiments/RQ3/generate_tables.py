#!/usr/bin/env python3
"""Generate the VLA-stratified RQ3 publication tables and statistical report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from experiments.RQ1.analysis import holm_adjust, wilcoxon_signed_rank
from experiments.RQ2.analysis import spearman_rank_correlation
from experiments.RQ3.analysis import matched_coverage_bootstrap


VOXEL_SIZE = 0.02
REFERENCE_FORMULA = "harmonic_mean"
EXPECTED_TECHNIQUES = 31
EXPECTED_PRIMARY_BLOCKS = 16
EXPECTED_CONFIGURATIONS = 32
EXPECTED_RUNS = 10
EXPECTED_CELLS = 4_800
PRACTICAL_MARGIN = 0.01
TRAINING_SIZES = (10, 20, 50, 100, 200)
COVERAGE_TIERS = ("low", "medium", "high")
VLA_ORDER = ("EO1", "Groot", "Pi0", "SpatialVLA")
VLA_DISPLAY = {
    "EO1": "EO1",
    "Groot": "GR00T",
    "Pi0": "Pi0",
    "SpatialVLA": "SpatialVLA",
}
TIER_DISPLAY = {"low": "Low", "medium": "Medium", "high": "High"}
SOURCE_FILES = (
    "cell_metrics.csv.gz",
    "matched_high_low_cells.csv.gz",
    "primary_block_metrics.csv",
    "rankings_by_coverage.csv",
    "statistical_tests.json",
    "summary.json",
    "technique_coverage_interactions.csv",
)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    root = repository_root()
    result_root = root / "experiments" / "RQ3" / "results" / "phase_a_coverage"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aggregate-dir",
        type=Path,
        default=result_root / "aggregate",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=result_root / "tables",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261024)
    parser.add_argument("--top-rank-cutoff", type=int, default=5)
    return parser.parse_args()


def read_csv(path: Path, required: Iterable[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(
            "{} is missing columns: {}".format(path, ", ".join(sorted(missing))))
    if frame.empty:
        raise ValueError("{} contains no rows".format(path))
    return frame


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("{} must contain a JSON object".format(path))
    return value


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_or_absolute(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repository_root()))
    except ValueError:
        return str(path.resolve())


def latex_escape(value: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "\\": r"\textbackslash{}",
    }
    return "".join(replacements.get(character, character) for character in value)


def bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not pd.isna(value) and value in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ValueError("invalid Boolean value: {!r}".format(value))


def validate_dimensions(
    cells: pd.DataFrame,
    matched: pd.DataFrame,
    primary: pd.DataFrame,
    rankings: pd.DataFrame,
    interactions: pd.DataFrame,
    summary: Mapping[str, object],
) -> None:
    if summary.get("status") != "complete":
        raise ValueError("RQ3 summary is not complete")
    if int(summary.get("completed_cells", -1)) != EXPECTED_CELLS:
        raise ValueError("RQ3 summary does not report 4,800 completed cells")
    for name, frame in (
        ("cell metrics", cells),
        ("matched cells", matched),
        ("primary blocks", primary),
        ("rankings", rankings),
    ):
        sizes = {float(value) for value in frame["voxel_size"].unique()}
        if sizes != {VOXEL_SIZE}:
            raise ValueError("{} must contain only voxel size 0.02".format(name))
    if set(primary["vla"].astype(str)) != set(VLA_ORDER):
        raise ValueError("primary blocks do not contain the four expected VLAs")
    if set(primary["training_size"].astype(int)) != set(TRAINING_SIZES):
        raise ValueError("primary blocks do not contain the five training sizes")
    if set(primary["coverage_tier"].astype(str)) != set(COVERAGE_TIERS):
        raise ValueError("primary blocks do not contain the three coverage tiers")

    formula_names = primary[["formula_id", "technique"]].drop_duplicates()
    if formula_names["formula_id"].nunique() != EXPECTED_TECHNIQUES:
        raise ValueError("primary blocks do not contain 31 techniques")
    if formula_names.groupby("formula_id")["technique"].nunique().max() != 1:
        raise ValueError("a formula identifier maps to multiple technique names")

    primary_counts = primary.groupby(
        ["vla", "condition", "training_size", "coverage_tier", "formula_id"]
    ).size()
    expected_primary = (
        len(VLA_ORDER) * 4 * len(TRAINING_SIZES)
        * len(COVERAGE_TIERS) * EXPECTED_TECHNIQUES
    )
    if len(primary_counts) != expected_primary or set(primary_counts.astype(int)) != {1}:
        raise ValueError("primary-block design is incomplete or duplicated")
    if primary["primary_block_id"].nunique() != EXPECTED_PRIMARY_BLOCKS:
        raise ValueError("expected 16 primary VLA-condition blocks")

    subset_keys = [
        "configuration_id", "test_run", "training_size",
        "selection_replicate", "coverage_tier",
    ]
    subset_cells = cells.drop_duplicates(subset_keys)
    if len(subset_cells) != EXPECTED_CELLS:
        raise ValueError("cell metrics do not contain 4,800 unique fitted subsets")
    if subset_cells["configuration_id"].nunique() != EXPECTED_CONFIGURATIONS:
        raise ValueError("cell metrics do not contain 32 configurations")
    if not (
        subset_cells["training_failures"].astype(int)
        == subset_cells["training_passes"].astype(int)
    ).all():
        raise ValueError("one or more training subsets are not outcome-balanced")
    if not (
        subset_cells["training_failures"].astype(int) * 2
        == subset_cells["training_size"].astype(int)
    ).all():
        raise ValueError("training subset class totals do not match training size")

    matched_keys = [
        "configuration_id", "test_run", "training_size",
        "selection_replicate", "formula_id",
    ]
    if matched.duplicated(matched_keys).any():
        raise ValueError("matched coverage cells contain duplicate keys")
    expected_matched = (
        EXPECTED_CONFIGURATIONS * EXPECTED_RUNS
        * len(TRAINING_SIZES) * EXPECTED_TECHNIQUES
    )
    if len(matched) != expected_matched:
        raise ValueError("matched coverage-cell design is incomplete")
    if len(rankings) != len(COVERAGE_TIERS) * EXPECTED_TECHNIQUES:
        raise ValueError("global coverage rankings are incomplete")
    if len(interactions) != EXPECTED_TECHNIQUES:
        raise ValueError("technique interaction results are incomplete")


def coverage_manipulation_data(
    cells: pd.DataFrame,
    primary: pd.DataFrame,
) -> pd.DataFrame:
    reference = primary[primary["formula_id"] == REFERENCE_FORMULA]
    means = reference.groupby(
        ["vla", "training_size", "coverage_tier"], as_index=False
    ).agg(
        calibration_unique_coverage=("calibration_unique_coverage", "mean"),
        test_unique_coverage=("test_unique_coverage", "mean"),
        primary_blocks=("primary_block_id", "nunique"),
    )
    raw_reference = cells[cells["formula_id"] == REFERENCE_FORMULA]
    raw = raw_reference.pivot(
        index=[
            "vla", "configuration_id", "test_run",
            "training_size", "selection_replicate",
        ],
        columns="coverage_tier",
        values=["calibration_unique_coverage", "test_unique_coverage"],
    )
    required = [
        (metric, tier)
        for metric in ("calibration_unique_coverage", "test_unique_coverage")
        for tier in COVERAGE_TIERS
    ]
    if any(column not in raw.columns for column in required) or raw[required].isna().any().any():
        raise ValueError("coverage-tier manipulation pairs are incomplete")
    if not (
        raw[("calibration_unique_coverage", "high")]
        > raw[("calibration_unique_coverage", "low")]
    ).all():
        raise ValueError("high calibration coverage did not exceed low coverage")
    transfer = (
        raw[("test_unique_coverage", "high")]
        > raw[("test_unique_coverage", "low")]
    ).rename("test_high_exceeds_low")
    transfer = transfer.reset_index().groupby(
        ["vla", "training_size"], as_index=False
    ).agg(
        test_high_exceeds_low_fraction=("test_high_exceeds_low", "mean"),
        matched_cells=("test_high_exceeds_low", "size"),
    )

    rows = []
    for vla in VLA_ORDER:
        for training_size in TRAINING_SIZES:
            selected = means[
                (means["vla"] == vla)
                & (means["training_size"].astype(int) == training_size)
            ].set_index("coverage_tier")
            if set(selected.index) != set(COVERAGE_TIERS):
                raise ValueError("coverage manipulation grid is incomplete")
            row = {
                "vla": vla,
                "vla_display": VLA_DISPLAY[vla],
                "training_size": training_size,
                "primary_blocks": int(selected["primary_blocks"].min()),
            }
            for metric in ("calibration_unique_coverage", "test_unique_coverage"):
                for tier in COVERAGE_TIERS:
                    row["{}_{}".format(metric, tier)] = float(selected.loc[tier, metric])
                row[metric + "_high_minus_low"] = (
                    row[metric + "_high"] - row[metric + "_low"])
            transfer_row = transfer[
                (transfer["vla"] == vla)
                & (transfer["training_size"].astype(int) == training_size)
            ].iloc[0]
            row["test_high_exceeds_low_fraction"] = float(
                transfer_row["test_high_exceeds_low_fraction"])
            row["matched_cells"] = int(transfer_row["matched_cells"])
            rows.append(row)
    return pd.DataFrame(rows)


def build_coverage_manipulation_table(
    data: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, Path]:
    csv_path = output_dir / "rq3_coverage_manipulation_by_vla.csv"
    write_csv(csv_path, data)
    latex_rows = []
    for vla in VLA_ORDER:
        latex_rows.append(
            r"\multicolumn{9}{l}{\textit{" + latex_escape(VLA_DISPLAY[vla]) + r"}} \\")
        group = data[data["vla"] == vla]
        for row in group.itertuples(index=False):
            latex_rows.append(
                "{} & {:.1f} & {:.1f} & {:.1f} & {:+.1f} & "
                "{:.1f} & {:.1f} & {:.1f} & {:+.1f} \\\\".format(
                    row.training_size,
                    100.0 * row.calibration_unique_coverage_low,
                    100.0 * row.calibration_unique_coverage_medium,
                    100.0 * row.calibration_unique_coverage_high,
                    100.0 * row.calibration_unique_coverage_high_minus_low,
                    100.0 * row.test_unique_coverage_low,
                    100.0 * row.test_unique_coverage_medium,
                    100.0 * row.test_unique_coverage_high,
                    100.0 * row.test_unique_coverage_high_minus_low,
                )
            )
        if vla != VLA_ORDER[-1]:
            latex_rows.append(r"\midrule")
    latex = "\n".join([
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{RQ3 -- Achieved unique-voxel coverage by VLA, training-set "
        r"size, and coverage tier. Calibration coverage selected the subsets; "
        r"test coverage was measured on untouched held-out trajectories. "
        r"$\Delta$ denotes high minus low in percentage points.}",
        r"\label{tab:rq3-coverage-manipulation}",
        r"{\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{rrrrrrrrr}",
        r"\toprule",
        r"& \multicolumn{4}{c}{Calibration coverage (\%)} "
        r"& \multicolumn{4}{c}{Test coverage (\%)} \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
        r"$n$ & Low & Medium & High & $\Delta$ & Low & Medium & High & $\Delta$ \\",
        r"\midrule",
        *latex_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table*}",
        "",
    ])
    tex_path = output_dir / "rq3_coverage_manipulation_by_vla.tex"
    atomic_text(tex_path, latex)
    return csv_path, tex_path


def reference_performance_data(primary: pd.DataFrame) -> pd.DataFrame:
    reference = primary[primary["formula_id"] == REFERENCE_FORMULA]
    vla = reference.groupby(
        ["vla", "training_size", "coverage_tier"], as_index=False
    ).agg(
        roc_auc=("roc_auc", "mean"),
        average_precision=("average_precision", "mean"),
        conditions=("condition", "nunique"),
    )
    macro = reference.groupby(
        ["training_size", "coverage_tier"], as_index=False
    ).agg(
        roc_auc=("roc_auc", "mean"),
        average_precision=("average_precision", "mean"),
        conditions=("primary_block_id", "nunique"),
    )
    macro["vla"] = "Macro"
    combined = pd.concat([vla, macro], ignore_index=True)
    rows = []
    for group_name in (*VLA_ORDER, "Macro"):
        for training_size in TRAINING_SIZES:
            selected = combined[
                (combined["vla"] == group_name)
                & (combined["training_size"].astype(int) == training_size)
            ].set_index("coverage_tier")
            if set(selected.index) != set(COVERAGE_TIERS):
                raise ValueError("reference performance grid is incomplete")
            row = {
                "vla": group_name,
                "vla_display": (
                    "Macro" if group_name == "Macro" else VLA_DISPLAY[group_name]),
                "training_size": training_size,
            }
            for tier in COVERAGE_TIERS:
                row["{}_roc_auc".format(tier)] = float(selected.loc[tier, "roc_auc"])
                row["{}_average_precision".format(tier)] = float(
                    selected.loc[tier, "average_precision"])
            row["high_minus_low_roc_auc"] = (
                row["high_roc_auc"] - row["low_roc_auc"])
            row["descriptive_best_tier"] = max(
                COVERAGE_TIERS, key=lambda tier: row[tier + "_roc_auc"])
            rows.append(row)
    return pd.DataFrame(rows)


def build_reference_performance_table(
    data: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, Path]:
    csv_path = output_dir / "rq3_reference_performance_by_vla.csv"
    write_csv(csv_path, data)
    latex_rows = []
    groups = (*VLA_ORDER, "Macro")
    for group_name in groups:
        display = "Macro" if group_name == "Macro" else VLA_DISPLAY[group_name]
        latex_rows.append(
            r"\multicolumn{5}{l}{\textit{" + latex_escape(display) + r"}} \\")
        group = data[data["vla"] == group_name]
        for row in group.itertuples(index=False):
            values = {
                tier: float(getattr(row, tier + "_roc_auc"))
                for tier in COVERAGE_TIERS
            }
            best = max(values.values())
            cells = []
            for tier in COVERAGE_TIERS:
                text = "{:.4f}".format(values[tier])
                if math.isclose(values[tier], best, rel_tol=0.0, abs_tol=1e-12):
                    text = r"\textbf{" + text + "}"
                cells.append(text)
            latex_rows.append(
                "{} & {} & {} & {} & {:+.4f} \\\\".format(
                    row.training_size, *cells, row.high_minus_low_roc_auc))
        if group_name != groups[-1]:
            latex_rows.append(r"\midrule")
    latex = "\n".join([
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{RQ3 -- Held-out ROC-AUC of Harmonic Mean by VLA, "
        r"training-set size, and coverage tier. High-minus-low contrasts compare "
        r"outcome-balanced subsets containing the same number of trajectories. "
        r"Bold denotes the descriptive best tier within a row.}",
        r"\label{tab:rq3-reference-performance}",
        r"{\scriptsize",
        r"\setlength{\tabcolsep}{8pt}",
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"$n$ & Low & Medium & High & High--low \\",
        r"\midrule",
        *latex_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table*}",
        "",
    ])
    tex_path = output_dir / "rq3_reference_performance_by_vla.tex"
    atomic_text(tex_path, latex)
    return csv_path, tex_path


def technique_metrics(
    primary: pd.DataFrame,
    rankings: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    vla = primary.groupby(
        ["vla", "coverage_tier", "formula_id", "technique"], as_index=False
    ).agg(
        macro_roc_auc=("roc_auc", "mean"),
        macro_average_precision=("average_precision", "mean"),
        training_sizes=("training_size", "nunique"),
        conditions=("condition", "nunique"),
    )
    vla["rank"] = vla.groupby(["vla", "coverage_tier"])[
        "macro_roc_auc"
    ].rank(method="min", ascending=False).astype(int)
    global_metrics = primary.groupby(
        ["coverage_tier", "formula_id", "technique"], as_index=False
    ).agg(
        macro_roc_auc=("roc_auc", "mean"),
        macro_average_precision=("average_precision", "mean"),
        training_sizes=("training_size", "nunique"),
        primary_blocks=("primary_block_id", "nunique"),
    )
    global_metrics["rank"] = global_metrics.groupby("coverage_tier")[
        "macro_roc_auc"
    ].rank(method="min", ascending=False).astype(int)
    check = rankings.merge(
        global_metrics[
            ["coverage_tier", "formula_id", "macro_roc_auc"]
        ].rename(columns={"macro_roc_auc": "recomputed_macro_roc_auc"}),
        on=["coverage_tier", "formula_id"],
        validate="one_to_one",
    )
    if not (
        (check["macro_roc_auc"] - check["recomputed_macro_roc_auc"]).abs() < 1e-12
    ).all():
        raise ValueError("global rankings do not reconcile to primary blocks")
    return vla, global_metrics


def wide_technique_row(group: pd.DataFrame, formula_id: str) -> dict:
    selected = group[group["formula_id"] == formula_id]
    if len(selected) != len(COVERAGE_TIERS):
        raise ValueError("{} has an incomplete coverage grid".format(formula_id))
    row = {
        "formula_id": formula_id,
        "technique": str(selected.iloc[0]["technique"]),
    }
    for tier in COVERAGE_TIERS:
        cell = selected[selected["coverage_tier"] == tier].iloc[0]
        row[tier + "_roc_auc"] = float(cell["macro_roc_auc"])
        row[tier + "_rank"] = int(cell["rank"])
    row["high_minus_low_roc_auc"] = (
        row["high_roc_auc"] - row["low_roc_auc"])
    row["minimum_rank"] = min(row[tier + "_rank"] for tier in COVERAGE_TIERS)
    return row


def format_auc_rank(value: float, rank: int, cutoff: int) -> str:
    text = "{:.4f} ({})".format(value, rank)
    if rank == 1:
        return r"\textbf{" + text + "}"
    if rank <= cutoff:
        return r"\underline{" + text + "}"
    return text


def technique_latex_rows(rows: Sequence[Mapping[str, object]], cutoff: int) -> list[str]:
    result = []
    for row in rows:
        cells = [latex_escape(str(row["technique"]))]
        for tier in COVERAGE_TIERS:
            cells.append(format_auc_rank(
                float(row[tier + "_roc_auc"]),
                int(row[tier + "_rank"]),
                cutoff,
            ))
        cells.append("{:+.4f}".format(float(row["high_minus_low_roc_auc"])))
        result.append(" & ".join(cells) + r" \\")
    return result


def ordered_technique_rows(
    group: pd.DataFrame,
    formula_ids: Iterable[str],
) -> list[dict]:
    rows = [wide_technique_row(group, formula_id) for formula_id in formula_ids]
    return sorted(
        rows,
        key=lambda row: (
            int(row["minimum_rank"]),
            int(row["medium_rank"]),
            str(row["technique"]),
        ),
    )


def build_technique_sensitivity_table(
    vla_metrics: pd.DataFrame,
    global_metrics: pd.DataFrame,
    output_dir: Path,
    cutoff: int,
) -> tuple[Path, Path, list[str]]:
    global_top = set(global_metrics.loc[
        global_metrics["rank"] <= cutoff, "formula_id"
    ].astype(str))
    vla_winners = set(vla_metrics.loc[
        vla_metrics["rank"] == 1, "formula_id"
    ].astype(str))
    selected_ids = sorted(global_top | vla_winners)
    rows = []
    panels = []
    for vla in VLA_ORDER:
        group = vla_metrics[vla_metrics["vla"] == vla]
        ordered = ordered_technique_rows(group, selected_ids)
        panels.append(
            r"\multicolumn{5}{l}{\textit{" + latex_escape(VLA_DISPLAY[vla]) + r"}} \\")
        panels.extend(technique_latex_rows(ordered, cutoff))
        if vla != VLA_ORDER[-1]:
            panels.append(r"\midrule")
        for row in ordered:
            row["vla"] = vla
            row["vla_display"] = VLA_DISPLAY[vla]
            reasons = []
            if row["formula_id"] in global_top:
                reasons.append("global_top_{}".format(cutoff))
            if row["formula_id"] in vla_winners:
                reasons.append("vla_tier_winner")
            row["selection_reason"] = ";".join(reasons)
            rows.append(row)
    output = pd.DataFrame(rows)
    columns = ["vla", "vla_display", "formula_id", "technique"]
    for tier in COVERAGE_TIERS:
        columns.extend([tier + "_roc_auc", tier + "_rank"])
    columns.extend([
        "high_minus_low_roc_auc", "minimum_rank", "selection_reason"])
    output = output[columns]
    csv_path = output_dir / "rq3_technique_sensitivity_by_vla.csv"
    write_csv(csv_path, output)
    latex = "\n".join([
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{RQ3 -- Coverage sensitivity of competitive SBFL techniques "
        r"by VLA. Included techniques reached a global top-five rank at some "
        r"coverage tier or ranked first for a VLA--tier combination. Cells "
        r"report macro ROC-AUC and tie-aware rank.}",
        r"\label{tab:rq3-technique-sensitivity}",
        r"{\scriptsize",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Technique & Low ROC-AUC (rank) & Medium ROC-AUC (rank) "
        r"& High ROC-AUC (rank) & High--low \\",
        r"\midrule",
        *panels,
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table*}",
        "",
    ])
    tex_path = output_dir / "rq3_technique_sensitivity_by_vla.tex"
    atomic_text(tex_path, latex)
    return csv_path, tex_path, selected_ids


def fold_delta_array(
    matched: pd.DataFrame,
) -> tuple[np.ndarray, list[str], list[int], list[str]]:
    frame = matched[np.isclose(matched["voxel_size"], VOXEL_SIZE)].copy()
    frame["primary_block_id"] = (
        frame["vla"].astype(str) + "__" + frame["condition"].astype(str))
    grouped = frame.groupby(
        [
            "primary_block_id", "budget", "test_run",
            "training_size", "formula_id",
        ],
        as_index=False,
    ).agg(delta_roc_auc=("delta_roc_auc", "mean"))
    blocks = sorted(grouped["primary_block_id"].astype(str).unique())
    budgets = sorted(grouped["budget"].astype(int).unique())
    runs = sorted(
        grouped["test_run"].astype(str).unique(),
        key=lambda value: int(value.rsplit("_", 1)[-1]),
    )
    training_sizes = sorted(grouped["training_size"].astype(int).unique())
    formula_ids = sorted(grouped["formula_id"].astype(str).unique())
    if len(blocks) != EXPECTED_PRIMARY_BLOCKS:
        raise ValueError("fold delta array requires 16 blocks")
    if len(budgets) != 2 or len(runs) != EXPECTED_RUNS:
        raise ValueError("fold delta array requires two budgets and ten runs")
    if tuple(training_sizes) != TRAINING_SIZES or len(formula_ids) != EXPECTED_TECHNIQUES:
        raise ValueError("fold delta array has unexpected dimensions")
    index = pd.MultiIndex.from_product(
        [blocks, budgets, runs, training_sizes, formula_ids],
        names=[
            "primary_block_id", "budget", "test_run",
            "training_size", "formula_id",
        ],
    )
    series = grouped.set_index(index.names)["delta_roc_auc"].reindex(index)
    if series.isna().any():
        raise ValueError("fold delta array contains missing matched cells")
    values = series.to_numpy(dtype=np.float64).reshape(
        len(blocks), len(budgets), len(runs),
        len(training_sizes), len(formula_ids),
    )
    return values, formula_ids, training_sizes, blocks


def block_reference_effects(primary: pd.DataFrame) -> pd.DataFrame:
    reference = primary[primary["formula_id"] == REFERENCE_FORMULA]
    pivot = reference.pivot(
        index=["primary_block_id", "vla", "condition", "training_size"],
        columns="coverage_tier",
        values=["roc_auc", "calibration_unique_coverage", "test_unique_coverage"],
    )
    required = [
        (metric, tier)
        for metric in (
            "roc_auc", "calibration_unique_coverage", "test_unique_coverage")
        for tier in ("low", "high")
    ]
    if any(column not in pivot.columns for column in required) or pivot[required].isna().any().any():
        raise ValueError("reference high-low primary-block pairs are incomplete")
    output = pivot.index.to_frame(index=False)
    for metric in ("roc_auc", "calibration_unique_coverage", "test_unique_coverage"):
        output["delta_" + metric] = (
            pivot[(metric, "high")].to_numpy(dtype=np.float64)
            - pivot[(metric, "low")].to_numpy(dtype=np.float64)
        )
    return output


def p_value_text(value: float) -> str:
    if value < 0.001:
        return r"$<0.001$"
    return "{:.3f}".format(value)


def statistical_data(
    primary: pd.DataFrame,
    matched: pd.DataFrame,
    rankings: pd.DataFrame,
    source_interactions: pd.DataFrame,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    values, formula_ids, training_sizes, _ = fold_delta_array(matched)
    reference_index = formula_ids.index(REFERENCE_FORMULA)
    bootstrap = matched_coverage_bootstrap(
        values,
        reference_technique_index=reference_index,
        iterations=bootstrap_iterations,
        seed=bootstrap_seed,
    )
    effects = block_reference_effects(primary)
    grouped_matched = matched.copy()
    grouped_matched["primary_block_id"] = (
        grouped_matched["vla"].astype(str)
        + "__" + grouped_matched["condition"].astype(str))
    grouped_matched = grouped_matched[
        grouped_matched["formula_id"] == REFERENCE_FORMULA
    ].groupby(
        ["primary_block_id", "training_size"], as_index=False
    ).agg(delta_roc_auc=("delta_roc_auc", "mean"))
    reconciliation = effects.merge(
        grouped_matched,
        on=["primary_block_id", "training_size"],
        suffixes=("_primary", "_matched"),
        validate="one_to_one",
    )
    if not (
        (reconciliation["delta_roc_auc_primary"]
         - reconciliation["delta_roc_auc_matched"]).abs() < 1e-12
    ).all():
        raise ValueError("reference matched effects do not reconcile")

    size_tests = {}
    raw_p_values = {}
    for training_size in training_sizes:
        sample = effects.loc[
            effects["training_size"].astype(int) == training_size,
            "delta_roc_auc",
        ].to_numpy(dtype=np.float64)
        result = wilcoxon_signed_rank(sample, np.zeros_like(sample))
        size_tests[training_size] = result
        raw_p_values[str(training_size)] = result.p_value
    adjusted = holm_adjust(raw_p_values)

    rows = []
    reference_point = values.mean(axis=(0, 1, 2))[:, reference_index]
    for size_index, training_size in enumerate(training_sizes):
        selected = effects[
            effects["training_size"].astype(int) == training_size]
        result = size_tests[training_size]
        lower = float(bootstrap.effect_lower[size_index, reference_index])
        upper = float(bootstrap.effect_upper[size_index, reference_index])
        point = float(reference_point[size_index])
        if not math.isclose(
            point, float(selected["delta_roc_auc"].mean()),
            rel_tol=0.0, abs_tol=1e-12,
        ):
            raise ValueError("bootstrap and primary reference effects disagree")
        rows.append({
            "scope": "training_size",
            "training_size": training_size,
            "primary_blocks": EXPECTED_PRIMARY_BLOCKS,
            "delta_calibration_unique_coverage": float(
                selected["delta_calibration_unique_coverage"].mean()),
            "delta_test_unique_coverage": float(
                selected["delta_test_unique_coverage"].mean()),
            "delta_roc_auc": point,
            "roc_auc_ci_lower": lower,
            "roc_auc_ci_upper": upper,
            "wilcoxon_statistic": result.statistic,
            "wilcoxon_p_value": result.p_value,
            "holm_adjusted_p_value": adjusted[str(training_size)],
            "rank_biserial": result.rank_biserial,
            "nonzero_pairs": result.nonzero_pairs,
            "practically_equivalent": bool(
                lower >= -PRACTICAL_MARGIN and upper <= PRACTICAL_MARGIN),
            "multiplicity": "Holm across five secondary training-size tests",
        })
    overall_by_block = effects.groupby("primary_block_id", as_index=False).agg(
        delta_roc_auc=("delta_roc_auc", "mean"),
        delta_calibration_unique_coverage=(
            "delta_calibration_unique_coverage", "mean"),
        delta_test_unique_coverage=("delta_test_unique_coverage", "mean"),
    )
    overall_sample = overall_by_block["delta_roc_auc"].to_numpy(dtype=np.float64)
    overall_test = wilcoxon_signed_rank(
        overall_sample, np.zeros_like(overall_sample))
    overall_lower = float(
        bootstrap.overall_effect_lower[reference_index])
    overall_upper = float(
        bootstrap.overall_effect_upper[reference_index])
    overall_point = float(reference_point.mean())
    rows.append({
        "scope": "overall_primary",
        "training_size": np.nan,
        "primary_blocks": EXPECTED_PRIMARY_BLOCKS,
        "delta_calibration_unique_coverage": float(
            overall_by_block["delta_calibration_unique_coverage"].mean()),
        "delta_test_unique_coverage": float(
            overall_by_block["delta_test_unique_coverage"].mean()),
        "delta_roc_auc": overall_point,
        "roc_auc_ci_lower": overall_lower,
        "roc_auc_ci_upper": overall_upper,
        "wilcoxon_statistic": overall_test.statistic,
        "wilcoxon_p_value": overall_test.p_value,
        "holm_adjusted_p_value": overall_test.p_value,
        "rank_biserial": overall_test.rank_biserial,
        "nonzero_pairs": overall_test.nonzero_pairs,
        "practically_equivalent": bool(
            overall_lower >= -PRACTICAL_MARGIN
            and overall_upper <= PRACTICAL_MARGIN),
        "multiplicity": "single prespecified primary contrast",
    })
    reference_tests = pd.DataFrame(rows)

    point_overall = values.mean(axis=(0, 1, 2, 3))
    source = source_interactions.copy()
    source["meaningful_interaction"] = source[
        "meaningful_interaction"].map(bool_value)
    interaction_rows = []
    for formula_index, formula_id in enumerate(formula_ids):
        selected = source[source["formula_id"] == formula_id]
        if len(selected) != 1:
            raise ValueError("interaction source has a missing or duplicate formula")
        selected_row = selected.iloc[0]
        technique_effect = float(point_overall[formula_index])
        reference_effect = float(point_overall[reference_index])
        interaction = technique_effect - reference_effect
        if not math.isclose(
            interaction,
            float(selected_row["interaction_vs_harmonic_mean"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("interaction source does not reconcile to fold effects")
        interaction_rows.append({
            "formula_id": formula_id,
            "technique": str(selected_row["technique"]),
            "mean_high_minus_low_auc": technique_effect,
            "reference_high_minus_low_auc": reference_effect,
            "interaction_vs_harmonic_mean": interaction,
            "interaction_ci_lower": float(
                bootstrap.overall_interaction_lower[formula_index]),
            "interaction_ci_upper": float(
                bootstrap.overall_interaction_upper[formula_index]),
            "wilcoxon_p_value": float(selected_row["wilcoxon_p_value"]),
            "holm_adjusted_p_value": float(
                selected_row["holm_adjusted_p_value"]),
            "rank_biserial": float(selected_row["rank_biserial"]),
            "holm_significant": bool(
                formula_id != REFERENCE_FORMULA
                and float(selected_row["holm_adjusted_p_value"]) < 0.05),
            "meaningful_interaction": bool(
                selected_row["meaningful_interaction"]),
        })
    interaction_tests = pd.DataFrame(interaction_rows).sort_values(
        ["meaningful_interaction", "holm_adjusted_p_value",
         "interaction_vs_harmonic_mean"],
        ascending=[False, True, False],
        ignore_index=True,
    )

    ranking_pivot = rankings.pivot(
        index="formula_id", columns="coverage_tier", values="macro_roc_auc")
    if ranking_pivot[list(COVERAGE_TIERS)].isna().any().any():
        raise ValueError("ranking stability grid is incomplete")
    stability_rows = []
    for first, second in (
        ("low", "medium"),
        ("low", "high"),
        ("medium", "high"),
    ):
        stability_rows.append({
            "first_coverage_tier": first,
            "second_coverage_tier": second,
            "spearman_rank_correlation": spearman_rank_correlation(
                ranking_pivot[first], ranking_pivot[second]),
            "techniques": len(ranking_pivot),
        })
    stability = pd.DataFrame(stability_rows)
    non_reference = interaction_tests[
        interaction_tests["formula_id"] != REFERENCE_FORMULA]
    largest = non_reference.loc[
        non_reference["interaction_vs_harmonic_mean"].abs().idxmax()]
    metadata = {
        "bootstrap_iterations": bootstrap_iterations,
        "bootstrap_seed": bootstrap_seed,
        "primary_test_p_value": overall_test.p_value,
        "primary_test_rank_biserial": overall_test.rank_biserial,
        "primary_effect": overall_point,
        "primary_ci": [overall_lower, overall_upper],
        "holm_significant_interactions": int(
            non_reference["holm_significant"].sum()),
        "meaningful_interactions": int(
            non_reference["meaningful_interaction"].sum()),
        "largest_absolute_interaction": {
            "formula_id": str(largest["formula_id"]),
            "technique": str(largest["technique"]),
            "value": float(largest["interaction_vs_harmonic_mean"]),
        },
    }
    return reference_tests, interaction_tests, stability, metadata


def build_statistical_tables(
    reference_tests: pd.DataFrame,
    interaction_tests: pd.DataFrame,
    stability: pd.DataFrame,
    metadata: Mapping[str, object],
    output_dir: Path,
) -> list[Path]:
    outputs = []
    reference_path = output_dir / "rq3_reference_coverage_tests.csv"
    interaction_path = output_dir / "rq3_technique_coverage_tests.csv"
    stability_path = output_dir / "rq3_ranking_stability.csv"
    write_csv(reference_path, reference_tests)
    write_csv(interaction_path, interaction_tests)
    write_csv(stability_path, stability)
    outputs.extend([reference_path, interaction_path, stability_path])

    reference_rows = []
    ordered = pd.concat([
        reference_tests[reference_tests["scope"] == "training_size"].sort_values(
            "training_size"),
        reference_tests[reference_tests["scope"] == "overall_primary"],
    ])
    for row in ordered.itertuples(index=False):
        label = (
            "Overall" if row.scope == "overall_primary"
            else str(int(row.training_size))
        )
        p_value = (
            row.wilcoxon_p_value
            if row.scope == "overall_primary"
            else row.holm_adjusted_p_value
        )
        reference_rows.append(
            "{} & {:+.1f} & {:+.1f} & {:+.4f} [{:+.4f}, {:+.4f}] "
            "& {} & {:+.3f} & {} \\\\".format(
                label,
                100.0 * row.delta_calibration_unique_coverage,
                100.0 * row.delta_test_unique_coverage,
                row.delta_roc_auc,
                row.roc_auc_ci_lower,
                row.roc_auc_ci_upper,
                p_value_text(p_value),
                row.rank_biserial,
                "Yes" if bool_value(row.practically_equivalent) else "No",
            )
        )
    stability_rows = [
        "{}--{} & {:.3f} \\\\".format(
            TIER_DISPLAY[row.first_coverage_tier],
            TIER_DISPLAY[row.second_coverage_tier],
            row.spearman_rank_correlation,
        )
        for row in stability.itertuples(index=False)
    ]
    largest = dict(metadata["largest_absolute_interaction"])
    latex = "\n".join([
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{RQ3 -- Global matched statistical analysis over the 16 "
        r"VLA--trajectory-condition blocks. Panel (a) tested Harmonic Mean's "
        r"high-minus-low effect. The overall row was the primary test; the five "
        r"size-specific $p$-values received Holm correction. Equivalence required "
        r"the complete 95\% interval to lie within $\pm0.01$ ROC-AUC.}",
        r"\label{tab:rq3-global-statistics}",
        r"{\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"\multicolumn{7}{l}{\textit{(a) Harmonic Mean coverage effect}} \\",
        r"$n$ & $\Delta$Cal. (pp) & $\Delta$Test (pp) "
        r"& $\Delta$AUC [95\% CI] & $p$ & $r_{rb}$ & Equivalent \\",
        r"\midrule",
        *reference_rows,
        r"\bottomrule",
        r"\end{tabular}",
        "",
        r"\vspace{2mm}",
        r"\begin{tabular}{rrrrl}",
        r"\toprule",
        r"\multicolumn{5}{l}{\textit{(b) Technique--coverage interaction family}} \\",
        r"Comparisons & Holm significant & Meaningful & Largest $|\Delta\Delta|$ "
        r"& Technique \\",
        r"\midrule",
        "{} & {} & {} & {:.4f} & {} \\\\".format(
            EXPECTED_TECHNIQUES - 1,
            int(metadata["holm_significant_interactions"]),
            int(metadata["meaningful_interactions"]),
            abs(float(largest["value"])),
            latex_escape(str(largest["technique"])),
        ),
        r"\bottomrule",
        r"\end{tabular}",
        "",
        r"\vspace{2mm}",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"\multicolumn{2}{l}{\textit{(c) Ranking stability}} \\",
        r"Coverage-tier pair & Spearman $\rho$ \\",
        r"\midrule",
        *stability_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table*}",
        "",
    ])
    tex_path = output_dir / "rq3_global_statistics.tex"
    atomic_text(tex_path, latex)
    outputs.append(tex_path)

    interaction_rows = []
    for row in interaction_tests.itertuples(index=False):
        if row.formula_id == REFERENCE_FORMULA:
            continue
        interaction_rows.append(
            "{} & {:+.4f} & {:+.4f} [{:+.4f}, {:+.4f}] & {} & {:+.3f} & {} \\\\".format(
                latex_escape(str(row.technique)),
                row.mean_high_minus_low_auc,
                row.interaction_vs_harmonic_mean,
                row.interaction_ci_lower,
                row.interaction_ci_upper,
                p_value_text(row.holm_adjusted_p_value),
                row.rank_biserial,
                "Yes" if bool_value(row.meaningful_interaction) else "No",
            )
        )
    interaction_latex = "\n".join([
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Complete RQ3 technique--coverage interaction analysis. "
        r"$\Delta$AUC is the technique's mean high-minus-low effect and "
        r"$\Delta\Delta$AUC subtracts the Harmonic Mean effect. $p$-values "
        r"received one Holm correction across 30 comparisons.}",
        r"\label{tab:rq3-supplementary-interactions}",
        r"{\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Technique & $\Delta$AUC & $\Delta\Delta$AUC [95\% CI] "
        r"& Holm $p$ & $r_{rb}$ & Meaningful \\",
        r"\midrule",
        *interaction_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table*}",
        "",
    ])
    interaction_tex = output_dir / "rq3_supplementary_interactions.tex"
    atomic_text(interaction_tex, interaction_latex)
    outputs.append(interaction_tex)
    return outputs


def build_supplementary_rankings(
    vla_metrics: pd.DataFrame,
    output_dir: Path,
    cutoff: int,
) -> list[Path]:
    outputs = []
    for vla in VLA_ORDER:
        group = vla_metrics[vla_metrics["vla"] == vla]
        formula_ids = sorted(group["formula_id"].astype(str).unique())
        if len(formula_ids) != EXPECTED_TECHNIQUES:
            raise ValueError("{} does not contain all 31 techniques".format(vla))
        rows = ordered_technique_rows(group, formula_ids)
        output = pd.DataFrame(rows)
        output.insert(0, "vla", vla)
        output.insert(1, "vla_display", VLA_DISPLAY[vla])
        csv_path = output_dir / "rq3_supplementary_rankings_{}.csv".format(
            VLA_DISPLAY[vla])
        write_csv(csv_path, output)
        outputs.append(csv_path)
        latex = "\n".join([
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{Complete RQ3 coverage-tier ranking for "
            + latex_escape(VLA_DISPLAY[vla])
            + r". Cells report macro ROC-AUC and tie-aware rank.}",
            r"\label{tab:rq3-supplementary-rankings-"
            + VLA_DISPLAY[vla].lower() + "}",
            r"{\scriptsize",
            r"\setlength{\tabcolsep}{5pt}",
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            r"Technique & Low ROC-AUC (rank) & Medium ROC-AUC (rank) "
            r"& High ROC-AUC (rank) & High--low \\",
            r"\midrule",
            *technique_latex_rows(rows, cutoff),
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table*}",
            "",
        ])
        tex_path = output_dir / "rq3_supplementary_rankings_{}.tex".format(
            VLA_DISPLAY[vla])
        atomic_text(tex_path, latex)
        outputs.append(tex_path)
    return outputs


def diagnostics_data(cells: pd.DataFrame) -> pd.DataFrame:
    reference = cells[cells["formula_id"] == REFERENCE_FORMULA]
    output = reference.groupby(
        ["vla", "training_size", "coverage_tier"], as_index=False
    ).agg(
        test_unique_coverage=("test_unique_coverage", "mean"),
        test_mean_known_fraction=("test_mean_known_fraction", "mean"),
        test_failure_known_fraction=("test_failure_known_fraction", "mean"),
        test_pass_known_fraction=("test_pass_known_fraction", "mean"),
        all_unknown_trajectories=("all_unknown_trajectories", "sum"),
        mean_training_voxels=("training_voxels", "mean"),
        mean_score_tie_fraction=("score_tie_fraction", "mean"),
        fitted_cells=("test_run", "size"),
    )
    output["vla_display"] = output["vla"].map(VLA_DISPLAY)
    order = {vla: index for index, vla in enumerate(VLA_ORDER)}
    tier_order = {tier: index for index, tier in enumerate(COVERAGE_TIERS)}
    output["_vla_order"] = output["vla"].map(order)
    output["_tier_order"] = output["coverage_tier"].map(tier_order)
    return output.sort_values(
        ["_vla_order", "training_size", "_tier_order"],
        ignore_index=True,
    ).drop(columns=["_vla_order", "_tier_order"])


def build_diagnostics_table(
    data: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, Path]:
    csv_path = output_dir / "rq3_supplementary_coverage_diagnostics.csv"
    write_csv(csv_path, data)
    latex_rows = []
    for vla in VLA_ORDER:
        latex_rows.append(
            r"\multicolumn{8}{l}{\textit{" + latex_escape(VLA_DISPLAY[vla]) + r"}} \\")
        group = data[data["vla"] == vla]
        for row in group.itertuples(index=False):
            latex_rows.append(
                "{} & {} & {:.1f} & {:.1f} & {:.1f} & {} & {:,.0f} & {:.1f} \\\\".format(
                    row.training_size,
                    TIER_DISPLAY[row.coverage_tier],
                    100.0 * row.test_mean_known_fraction,
                    100.0 * row.test_failure_known_fraction,
                    100.0 * row.test_pass_known_fraction,
                    int(row.all_unknown_trajectories),
                    row.mean_training_voxels,
                    100.0 * row.mean_score_tie_fraction,
                )
            )
        if vla != VLA_ORDER[-1]:
            latex_rows.append(r"\midrule")
    latex = "\n".join([
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{RQ3 coverage diagnostics by VLA, training-set size, and "
        r"coverage tier for Harmonic Mean. Known and tie values are percentages; "
        r"unknown is the total number of all-unknown test trajectories over 80 "
        r"matched fitted cells.}",
        r"\label{tab:rq3-supplementary-diagnostics}",
        r"{\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{rlrrrrrr}",
        r"\toprule",
        r"$n$ & Tier & Test known & Failure known & Pass known "
        r"& Unknown & Train voxels & Ties \\",
        r"\midrule",
        *latex_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table*}",
        "",
    ])
    tex_path = output_dir / "rq3_supplementary_coverage_diagnostics.tex"
    atomic_text(tex_path, latex)
    return csv_path, tex_path


def generate_tables(args: argparse.Namespace) -> list[Path]:
    if args.bootstrap_iterations < 100:
        raise ValueError("bootstrap iterations must be at least 100")
    if args.top_rank_cutoff < 1:
        raise ValueError("top-rank cutoff must be positive")
    aggregate_dir = args.aggregate_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cells = read_csv(
        aggregate_dir / "cell_metrics.csv.gz",
        (
            "voxel_size", "configuration_id", "vla", "condition", "budget",
            "test_run", "training_size", "coverage_tier",
            "selection_replicate", "formula_id", "technique", "roc_auc",
            "training_failures", "training_passes", "training_voxels",
            "calibration_unique_coverage", "test_unique_coverage",
            "test_mean_known_fraction", "test_failure_known_fraction",
            "test_pass_known_fraction", "all_unknown_trajectories",
            "score_tie_fraction",
        ),
    )
    matched = read_csv(
        aggregate_dir / "matched_high_low_cells.csv.gz",
        (
            "voxel_size", "configuration_id", "vla", "condition", "budget",
            "test_run", "training_size", "selection_replicate", "formula_id",
            "technique", "delta_roc_auc",
        ),
    )
    primary = read_csv(
        aggregate_dir / "primary_block_metrics.csv",
        (
            "voxel_size", "vla", "condition", "training_size",
            "coverage_tier", "formula_id", "technique", "roc_auc",
            "average_precision", "calibration_unique_coverage",
            "test_unique_coverage", "test_mean_known_fraction",
            "test_failure_known_fraction", "test_pass_known_fraction",
            "training_voxels", "all_unknown_trajectories",
            "score_tie_fraction", "primary_block_id",
        ),
    )
    rankings = read_csv(
        aggregate_dir / "rankings_by_coverage.csv",
        (
            "voxel_size", "coverage_tier", "formula_id", "technique",
            "macro_roc_auc", "rank",
        ),
    )
    interactions = read_csv(
        aggregate_dir / "technique_coverage_interactions.csv",
        (
            "formula_id", "technique", "mean_high_minus_low_auc",
            "reference_high_minus_low_auc",
            "interaction_vs_harmonic_mean", "wilcoxon_p_value",
            "holm_adjusted_p_value", "rank_biserial",
            "meaningful_interaction",
        ),
    )
    summary = read_json(aggregate_dir / "summary.json")
    statistical_source = read_json(aggregate_dir / "statistical_tests.json")
    if statistical_source.get("status") != "complete":
        raise ValueError("source interaction analysis is not complete")
    validate_dimensions(
        cells, matched, primary, rankings, interactions, summary)

    outputs = []
    coverage = coverage_manipulation_data(cells, primary)
    outputs.extend(build_coverage_manipulation_table(coverage, output_dir))
    reference = reference_performance_data(primary)
    outputs.extend(build_reference_performance_table(reference, output_dir))
    vla_metrics, global_metrics = technique_metrics(primary, rankings)
    competitive_csv, competitive_tex, selected_ids = (
        build_technique_sensitivity_table(
            vla_metrics, global_metrics, output_dir, args.top_rank_cutoff))
    outputs.extend([competitive_csv, competitive_tex])

    reference_tests, interaction_tests, stability, statistics = statistical_data(
        primary,
        matched,
        rankings,
        interactions,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
    )
    outputs.extend(build_statistical_tables(
        reference_tests, interaction_tests, stability, statistics, output_dir))
    outputs.extend(build_supplementary_rankings(
        vla_metrics, output_dir, args.top_rank_cutoff))
    diagnostics = diagnostics_data(cells)
    outputs.extend(build_diagnostics_table(diagnostics, output_dir))

    manifest = {
        "source_directory": relative_or_absolute(aggregate_dir),
        "source_sha256": {
            filename: sha256_file(aggregate_dir / filename)
            for filename in SOURCE_FILES
        },
        "voxel_size": VOXEL_SIZE,
        "training_sizes": list(TRAINING_SIZES),
        "coverage_tiers": list(COVERAGE_TIERS),
        "vla_order": [VLA_DISPLAY[vla] for vla in VLA_ORDER],
        "technique_count": EXPECTED_TECHNIQUES,
        "reference_formula": REFERENCE_FORMULA,
        "primary_blocks": EXPECTED_PRIMARY_BLOCKS,
        "competitive_selection_rule": (
            "global top-five at any coverage tier or rank one "
            "for any VLA-tier combination"),
        "competitive_formula_ids": selected_ids,
        "statistical_scope": "global over 16 matched VLA-condition blocks",
        "descriptive_scope": "stratified by VLA and training-set size",
        "bootstrap": {
            "method": (
                "matched hierarchical resampling of VLA-condition blocks and "
                "run folds within equal-weight budgets; identical draws across "
                "training sizes and techniques"),
            "iterations": args.bootstrap_iterations,
            "seed": args.bootstrap_seed,
        },
        "practical_equivalence_margin": PRACTICAL_MARGIN,
        "statistics": statistics,
        "outputs": [path.name for path in outputs],
    }
    manifest_path = output_dir / "manifest.json"
    atomic_text(
        manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    outputs.append(manifest_path)
    return outputs


def main() -> None:
    args = parse_args()
    outputs = generate_tables(args)
    print(
        "Generated {} RQ3 table artifacts in {}".format(
            len(outputs), args.output_dir.resolve()))
    for path in outputs:
        print("- {}".format(path.resolve()))


if __name__ == "__main__":
    main()
