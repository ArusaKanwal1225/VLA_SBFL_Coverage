#!/usr/bin/env python3
"""Generate VLA-stratified publication tables from the RQ2 results.

The matched inferential analysis remains owned by ``run_experiment.py``. Run it
with ``--analysis-only`` to recompute the statistical artifacts, then run this
script to validate them and produce CSV and LaTeX tables for the paper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import pandas as pd


VOXEL_SIZES = (0.01, 0.02, 0.05)
PRIMARY_SIZE = 0.02
REFERENCE_FORMULA = "harmonic_mean"
EXPECTED_TECHNIQUES = 34
EXPECTED_CONDITIONS = 4
EXPECTED_FOLDS_PER_VLA_SIZE = 80
VLA_ORDER = ("EO1", "Groot", "Pi0", "SpatialVLA")
VLA_DISPLAY = {
    "EO1": "EO1",
    "Groot": "GR00T",
    "Pi0": "Pi0",
    "SpatialVLA": "SpatialVLA",
}
SIZE_LABEL = {0.01: "0.01", 0.02: "0.02", 0.05: "0.05"}
SIZE_FIELD = {0.01: "0p01", 0.02: "0p02", 0.05: "0p05"}
AGGREGATE_INPUTS = (
    "fold_metrics_by_size.csv.gz",
    "primary_block_metrics.csv",
    "rankings_by_size.csv",
    "ranking_stability.csv",
    "size_diagnostics.csv",
    "statistical_tests.json",
    "technique_size_effects.csv",
    "technique_size_interactions.csv",
    "voxel_size_comparison.csv",
)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    root = repository_root()
    rq2_root = root / "experiments" / "RQ2" / "results" / "phase_a_granularity"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-dir", type=Path, default=rq2_root / "aggregate")
    parser.add_argument("--fine-results", type=Path, default=rq2_root / "cells" / "size_0p01")
    parser.add_argument(
        "--primary-results",
        type=Path,
        default=root / "experiments" / "RQ1" / "results" / "phase_a_primary",
    )
    parser.add_argument("--coarse-results", type=Path, default=rq2_root / "cells" / "size_0p05")
    parser.add_argument("--output-dir", type=Path, default=rq2_root / "tables")
    parser.add_argument(
        "--top-rank-cutoff",
        type=int,
        default=5,
        help="Include a technique in a VLA panel if it reaches this rank at any size.",
    )
    return parser.parse_args()


def read_csv(path: Path, required: Iterable[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError("{} is missing columns: {}".format(path, ", ".join(sorted(missing))))
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
    if isinstance(value, (int, float)) and not pd.isna(value):
        if value in (0, 1):
            return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ValueError("invalid Boolean value: {!r}".format(value))


def size_matches(value: float, expected: float) -> bool:
    return math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-12)


def relative_or_absolute(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repository_root()))
    except ValueError:
        return str(path.resolve())


def validate_experiment_dimensions(
    primary: pd.DataFrame,
    folds: pd.DataFrame,
    rankings: pd.DataFrame,
) -> None:
    expected_vlas = set(VLA_ORDER)
    for name, frame in (("primary blocks", primary), ("fold metrics", folds)):
        if set(frame["vla"].astype(str)) != expected_vlas:
            raise ValueError("{} do not contain exactly the four expected VLAs".format(name))
        sizes = {float(value) for value in frame["voxel_size"].unique()}
        if sizes != set(VOXEL_SIZES):
            raise ValueError("{} do not contain exactly the three voxel sizes".format(name))

    formula_names = primary[["formula_id", "technique"]].drop_duplicates()
    if formula_names["formula_id"].nunique() != EXPECTED_TECHNIQUES:
        raise ValueError("primary blocks do not contain 34 techniques")
    if formula_names.groupby("formula_id")["technique"].nunique().max() != 1:
        raise ValueError("a formula identifier maps to multiple technique names")

    primary_counts = primary.groupby(["vla", "voxel_size", "formula_id"])[
        "condition"
    ].nunique()
    if len(primary_counts) != len(VLA_ORDER) * len(VOXEL_SIZES) * EXPECTED_TECHNIQUES:
        raise ValueError("primary-block design is incomplete")
    if set(primary_counts.astype(int)) != {EXPECTED_CONDITIONS}:
        raise ValueError("each VLA-size-technique cell must contain four conditions")

    reference_folds = folds[folds["formula_id"] == REFERENCE_FORMULA]
    fold_counts = reference_folds.groupby(["vla", "voxel_size"]).size()
    if len(fold_counts) != len(VLA_ORDER) * len(VOXEL_SIZES):
        raise ValueError("reference fold design is incomplete")
    if set(fold_counts.astype(int)) != {EXPECTED_FOLDS_PER_VLA_SIZE}:
        raise ValueError("each VLA-size cell must contain 80 reference folds")

    ranking_counts = rankings.groupby("voxel_size")["formula_id"].nunique()
    if len(ranking_counts) != len(VOXEL_SIZES) or set(ranking_counts.astype(int)) != {
        EXPECTED_TECHNIQUES
    }:
        raise ValueError("rankings_by_size.csv is incomplete")


def aggregate_vla_metrics(primary: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
    metrics = primary.groupby(
        ["vla", "voxel_size", "formula_id", "technique"], as_index=False
    ).agg(
        macro_roc_auc=("roc_auc", "mean"),
        macro_average_precision=("average_precision", "mean"),
        mean_known_fraction=("mean_known_fraction", "mean"),
        mean_score_tie_fraction=("score_tie_fraction", "mean"),
        conditions=("condition", "nunique"),
    )
    metrics["within_vla_size_rank"] = metrics.groupby(["vla", "voxel_size"])[
        "macro_roc_auc"
    ].rank(method="min", ascending=False).astype(int)

    recomputed = primary.groupby(
        ["voxel_size", "formula_id"], as_index=False
    )["roc_auc"].mean().rename(columns={"roc_auc": "recomputed_macro_roc_auc"})
    check = rankings.merge(
        recomputed, on=["voxel_size", "formula_id"], validate="one_to_one"
    )
    if not (
        (check["macro_roc_auc"] - check["recomputed_macro_roc_auc"]).abs() < 1e-12
    ).all():
        raise ValueError("rankings_by_size.csv does not reconcile to primary blocks")
    return metrics


def auc_rank_cell(value: float, rank: int, cutoff: int) -> str:
    text = "{:.4f} ({})".format(value, rank)
    if rank == 1:
        return r"\textbf{" + text + "}"
    if rank <= cutoff:
        return r"\underline{" + text + "}"
    return text


def build_reference_table(
    vla_metrics: pd.DataFrame,
    rankings: pd.DataFrame,
    output_dir: Path,
) -> Tuple[Path, Path]:
    rows: List[Dict[str, object]] = []
    latex_rows: List[str] = []
    reference = vla_metrics[vla_metrics["formula_id"] == REFERENCE_FORMULA]
    groups: List[Tuple[str, pd.DataFrame]] = [
        (vla, reference[reference["vla"] == vla]) for vla in VLA_ORDER
    ]
    macro = rankings[rankings["formula_id"] == REFERENCE_FORMULA].copy()
    groups.append(("Macro", macro))

    for group, frame in groups:
        if len(frame) != len(VOXEL_SIZES):
            raise ValueError("{} reference table has an incomplete size grid".format(group))
        values = {
            size: float(
                frame.loc[frame["voxel_size"].map(lambda value: size_matches(value, size)), "macro_roc_auc"].iloc[0]
            )
            for size in VOXEL_SIZES
        }
        best = max(values.values())
        best_sizes = [size for size in VOXEL_SIZES if math.isclose(values[size], best, abs_tol=1e-12)]
        display = "Macro" if group == "Macro" else VLA_DISPLAY[group]
        row: Dict[str, object] = {"vla": display}
        cells = [r"\textbf{Macro}" if group == "Macro" else latex_escape(display)]
        for size in VOXEL_SIZES:
            field = "size_{}_roc_auc".format(SIZE_FIELD[size])
            row[field] = values[size]
            formatted = "{:.4f}".format(values[size])
            cells.append(r"\textbf{" + formatted + "}" if size in best_sizes else formatted)
        row["descriptive_best_size"] = ";".join(SIZE_LABEL[size] for size in best_sizes)
        cells.append(" / ".join(SIZE_LABEL[size] for size in best_sizes))
        rows.append(row)
        latex_rows.append(" & ".join(cells) + r" \\")

    frame = pd.DataFrame(rows)
    csv_path = output_dir / "rq2_reference_by_vla.csv"
    write_csv(csv_path, frame)
    latex = "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{Voxel-size sensitivity of Harmonic Mean, the frozen RQ1 "
            r"reference technique. Values are macro ROC-AUC over the four "
            r"trajectory-generation conditions for each VLA. Bold denotes the "
            r"descriptive best size within a row; inferential tests are reported "
            r"separately in Table~\ref{tab:rq2-global-statistics}.}",
            r"\label{tab:rq2-reference-by-vla}",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"VLA & 0.01 & 0.02 & 0.05 & Descriptive best size \\",
            r"\midrule",
            *latex_rows[:-1],
            r"\midrule",
            latex_rows[-1],
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    tex_path = output_dir / "rq2_reference_by_vla.tex"
    atomic_text(tex_path, latex)
    return csv_path, tex_path


def technique_wide_row(group: pd.DataFrame, formula_id: str) -> Dict[str, object]:
    selected = group[group["formula_id"] == formula_id]
    if len(selected) != len(VOXEL_SIZES):
        raise ValueError("{} has an incomplete size grid".format(formula_id))
    first = selected.iloc[0]
    row: Dict[str, object] = {
        "formula_id": formula_id,
        "technique": str(first["technique"]),
    }
    for size in VOXEL_SIZES:
        cell = selected[selected["voxel_size"].map(lambda value: size_matches(value, size))].iloc[0]
        suffix = SIZE_FIELD[size]
        row["size_{}_roc_auc".format(suffix)] = float(cell["macro_roc_auc"])
        row["size_{}_rank".format(suffix)] = int(cell["within_vla_size_rank"])
    row["minimum_rank"] = min(int(row["size_{}_rank".format(SIZE_FIELD[size])]) for size in VOXEL_SIZES)
    return row


def ordered_technique_rows(group: pd.DataFrame, formula_ids: Iterable[str]) -> List[Dict[str, object]]:
    rows = [technique_wide_row(group, formula_id) for formula_id in formula_ids]
    return sorted(
        rows,
        key=lambda row: (
            int(row["minimum_rank"]),
            int(row["size_0p02_rank"]),
            str(row["technique"]),
        ),
    )


def latex_technique_rows(rows: Sequence[Mapping[str, object]], cutoff: int) -> List[str]:
    result = []
    for row in rows:
        cells = [latex_escape(str(row["technique"]))]
        for size in VOXEL_SIZES:
            suffix = SIZE_FIELD[size]
            cells.append(
                auc_rank_cell(
                    float(row["size_{}_roc_auc".format(suffix)]),
                    int(row["size_{}_rank".format(suffix)]),
                    cutoff,
                )
            )
        result.append(" & ".join(cells) + r" \\")
    return result


def build_competitive_technique_table(
    vla_metrics: pd.DataFrame,
    output_dir: Path,
    cutoff: int,
) -> Tuple[Path, Path, Dict[str, List[str]]]:
    all_rows: List[Dict[str, object]] = []
    panels: List[str] = []
    included: Dict[str, List[str]] = {}
    for panel_index, vla in enumerate(VLA_ORDER):
        group = vla_metrics[vla_metrics["vla"] == vla]
        formula_ids = sorted(
            set(group.loc[group["within_vla_size_rank"] <= cutoff, "formula_id"].astype(str))
        )
        included[vla] = formula_ids
        rows = ordered_technique_rows(group, formula_ids)
        for row in rows:
            row["vla"] = VLA_DISPLAY[vla]
            row["top_rank_cutoff"] = cutoff
            all_rows.append(row)
        letter = chr(ord("a") + panel_index)
        panels.append(
            r"\multicolumn{4}{l}{\textit{(" + letter + ") " + latex_escape(VLA_DISPLAY[vla]) + r"}} \\"
        )
        panels.extend(latex_technique_rows(rows, cutoff))
        if vla != VLA_ORDER[-1]:
            panels.append(r"\midrule")

    output = pd.DataFrame(all_rows)
    columns = ["vla", "formula_id", "technique"]
    for size in VOXEL_SIZES:
        columns.extend(
            [
                "size_{}_roc_auc".format(SIZE_FIELD[size]),
                "size_{}_rank".format(SIZE_FIELD[size]),
            ]
        )
    columns.extend(["minimum_rank", "top_rank_cutoff"])
    output = output[columns]
    csv_path = output_dir / "rq2_techniques_by_vla.csv"
    write_csv(csv_path, output)

    latex = "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{Voxel-size sensitivity of techniques reaching a top-five "
            r"rank for at least one size within each VLA. Each cell reports macro "
            r"ROC-AUC and the rank in parentheses. Ranks use unrounded values. Bold "
            r"denotes rank 1 and underlining denotes the remaining top-five ranks.}",
            r"\label{tab:rq2-techniques-by-vla}",
            r"{\scriptsize",
            r"\setlength{\tabcolsep}{7pt}",
            r"\renewcommand{\arraystretch}{0.96}",
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"Technique & 0.01 ROC-AUC (rank) & 0.02 ROC-AUC (rank) & 0.05 ROC-AUC (rank) \\",
            r"\midrule",
            *panels,
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table*}",
            "",
        ]
    )
    tex_path = output_dir / "rq2_techniques_by_vla.tex"
    atomic_text(tex_path, latex)
    return csv_path, tex_path, included


def load_training_voxels(sources: Mapping[float, Path]) -> pd.DataFrame:
    rows = []
    for size in VOXEL_SIZES:
        source = sources[size].expanduser().resolve()
        metadata_paths = sorted((source / "folds").glob("*/*/model/metadata.json"))
        if len(metadata_paths) != len(VLA_ORDER) * EXPECTED_FOLDS_PER_VLA_SIZE:
            raise ValueError(
                "{} expected 320 model metadata files, found {}".format(
                    source, len(metadata_paths)
                )
            )
        for path in metadata_paths:
            metadata = read_json(path)
            grid = metadata.get("grid", {})
            training = metadata.get("training", {})
            if not size_matches(float(grid.get("voxel_size", math.nan)), size):
                raise ValueError("{} has an unexpected voxel size".format(path))
            rows.append(
                {
                    "voxel_size": size,
                    "configuration_id": str(training["configuration_id"]),
                    "test_run": str(training["test_run"]),
                    "vla": str(training["vla"]),
                    "training_voxels": int(training["training_voxels"]),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.duplicated(["voxel_size", "configuration_id", "test_run"]).any():
        raise ValueError("training voxel metadata contains duplicate folds")
    if set(frame["vla"]) != set(VLA_ORDER):
        raise ValueError("training voxel metadata contains unexpected VLAs")
    return frame


def build_diagnostics_table(
    folds: pd.DataFrame,
    training_voxels: pd.DataFrame,
    global_diagnostics: pd.DataFrame,
    output_dir: Path,
) -> Tuple[Path, Path]:
    reference = folds[folds["formula_id"] == REFERENCE_FORMULA].copy()
    diagnostics = reference.groupby(["vla", "voxel_size"], as_index=False).agg(
        mean_known_fraction=("mean_known_fraction", "mean"),
        all_unknown_trajectories=("all_unknown_trajectories", "sum"),
        mean_score_tie_fraction=("score_tie_fraction", "mean"),
        mean_fit_seconds=("fit_seconds", "mean"),
        mean_score_seconds=("score_seconds", "mean"),
        mean_model_artifact_bytes=("model_artifact_bytes", "mean"),
        folds=("test_run", "size"),
    )
    voxel_summary = training_voxels.groupby(["vla", "voxel_size"], as_index=False).agg(
        mean_training_voxels=("training_voxels", "mean"),
        median_training_voxels=("training_voxels", "median"),
        minimum_training_voxels=("training_voxels", "min"),
        maximum_training_voxels=("training_voxels", "max"),
    )
    diagnostics = diagnostics.merge(
        voxel_summary, on=["vla", "voxel_size"], validate="one_to_one"
    )
    diagnostics["vla_display"] = diagnostics["vla"].map(VLA_DISPLAY)
    diagnostics["artifact_megabytes"] = diagnostics["mean_model_artifact_bytes"] / 1_000_000.0
    diagnostics["vla_order"] = diagnostics["vla"].map(
        {vla: index for index, vla in enumerate(VLA_ORDER)}
    )
    diagnostics = diagnostics.sort_values(["vla_order", "voxel_size"], ignore_index=True)

    global_reference = reference.groupby("voxel_size", as_index=False).agg(
        mean_known_fraction=("mean_known_fraction", "mean"),
        all_unknown_trajectories=("all_unknown_trajectories", "sum"),
        mean_score_tie_fraction=("score_tie_fraction", "mean"),
        mean_fit_seconds=("fit_seconds", "mean"),
        mean_score_seconds=("score_seconds", "mean"),
        mean_model_artifact_bytes=("model_artifact_bytes", "mean"),
        folds=("test_run", "size"),
    ).merge(
        training_voxels.groupby("voxel_size", as_index=False).agg(
            mean_training_voxels=("training_voxels", "mean")
        ),
        on="voxel_size",
        validate="one_to_one",
    )
    fields = [
        "mean_known_fraction",
        "all_unknown_trajectories",
        "mean_score_tie_fraction",
        "mean_fit_seconds",
        "mean_score_seconds",
        "mean_model_artifact_bytes",
        "folds",
        "mean_training_voxels",
    ]
    check = global_diagnostics.merge(
        global_reference, on="voxel_size", suffixes=("_source", "_recomputed"), validate="one_to_one"
    )
    for field in fields:
        left = check["{}_source".format(field)].astype(float)
        right = check["{}_recomputed".format(field)].astype(float)
        if not ((left - right).abs() < 1e-9).all():
            raise ValueError("VLA diagnostics do not reconcile for {}".format(field))

    csv_columns = [
        "vla",
        "vla_display",
        "voxel_size",
        "mean_known_fraction",
        "all_unknown_trajectories",
        "mean_score_tie_fraction",
        "mean_training_voxels",
        "median_training_voxels",
        "minimum_training_voxels",
        "maximum_training_voxels",
        "mean_model_artifact_bytes",
        "artifact_megabytes",
        "mean_fit_seconds",
        "mean_score_seconds",
        "folds",
    ]
    csv_path = output_dir / "rq2_diagnostics_by_vla.csv"
    write_csv(csv_path, diagnostics[csv_columns])

    latex_rows = []
    for vla in VLA_ORDER:
        group = diagnostics[diagnostics["vla"] == vla]
        for index, row in enumerate(group.itertuples(index=False)):
            display = latex_escape(VLA_DISPLAY[vla]) if index == 0 else ""
            latex_rows.append(
                r"{} & {:.2f} & {:.1f}\% & {:.1f}\% & {:,.0f} & {:.2f} & {:.3f} & {:.3f} \\".format(
                    display,
                    row.voxel_size,
                    100.0 * row.mean_known_fraction,
                    100.0 * row.mean_score_tie_fraction,
                    row.mean_training_voxels,
                    row.artifact_megabytes,
                    row.mean_fit_seconds,
                    row.mean_score_seconds,
                )
            )
        if vla != VLA_ORDER[-1]:
            latex_rows.append(r"\midrule")
    latex = "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{Coverage and computational diagnostics by VLA and voxel "
            r"size for Harmonic Mean. Values are averaged over the 80 held-out "
            r"folds in each VLA-size cell. Artifact sizes use decimal megabytes.}",
            r"\label{tab:rq2-diagnostics-by-vla}",
            r"{\scriptsize",
            r"\setlength{\tabcolsep}{4pt}",
            r"\begin{tabular}{lrrrrrrr}",
            r"\toprule",
            r"VLA & Size & Known & Ties & Train voxels & Artifact (MB) & Fit (s) & Score (s) \\",
            r"\midrule",
            *latex_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table*}",
            "",
        ]
    )
    tex_path = output_dir / "rq2_diagnostics_by_vla.tex"
    atomic_text(tex_path, latex)
    return csv_path, tex_path


def p_value_text(value: float) -> str:
    if value < 0.001:
        return r"$<0.001$"
    return "{:.3f}".format(value)


def build_statistical_tables(
    size_comparison: pd.DataFrame,
    interactions: pd.DataFrame,
    effects: pd.DataFrame,
    stability: pd.DataFrame,
    tests: Mapping[str, object],
    output_dir: Path,
) -> Tuple[List[Path], int]:
    size_test = dict(tests.get("size_friedman", {}))
    required_test_fields = {
        "statistic",
        "degrees_of_freedom",
        "p_value",
        "primary_blocks",
        "posthoc_performed",
    }
    if not required_test_fields <= set(size_test):
        raise ValueError("statistical_tests.json has an incomplete Friedman result")
    if int(size_test["primary_blocks"]) != 16:
        raise ValueError("the global size test must use 16 primary blocks")
    if int(tests.get("interaction_holm_family", -1)) != 66:
        raise ValueError("the interaction Holm family must contain 66 contrasts")
    if int(tests.get("size_change_holm_family", -1)) != 68:
        raise ValueError("the size-change Holm family must contain 68 contrasts")

    size_output = size_comparison[
        [
            "voxel_size",
            "macro_roc_auc",
            "roc_auc_ci_lower",
            "roc_auc_ci_upper",
            "delta_auc_vs_primary_0p02",
            "delta_vs_primary_ci_lower",
            "delta_vs_primary_ci_upper",
            "practically_equivalent_to_size_leader",
            "phase_a_size_rank",
        ]
    ].sort_values("voxel_size").copy()
    size_output["friedman_statistic"] = float(size_test["statistic"])
    size_output["friedman_degrees_of_freedom"] = int(size_test["degrees_of_freedom"])
    size_output["friedman_p_value"] = float(size_test["p_value"])
    size_output["friedman_posthoc_performed"] = bool(size_test["posthoc_performed"])
    size_csv = output_dir / "rq2_global_size_test.csv"
    write_csv(size_csv, size_output)

    meaningful = interactions[
        interactions["practically_meaningful_interaction"].map(bool_value)
    ].copy()
    effect_columns = effects[
        [
            "voxel_size",
            "formula_id",
            "delta_macro_roc_auc",
            "delta_ci_lower",
            "delta_ci_upper",
            "holm_adjusted_p_value",
            "practically_meaningful_change",
        ]
    ].rename(
        columns={
            "holm_adjusted_p_value": "size_change_holm_adjusted_p_value",
            "practically_meaningful_change": "practically_meaningful_size_change",
        }
    )
    meaningful = meaningful.merge(
        effect_columns,
        on=["voxel_size", "formula_id"],
        validate="one_to_one",
    ).sort_values(
        ["voxel_size", "difference_in_differences_roc_auc"],
        ascending=[True, False],
        ignore_index=True,
    )
    interaction_csv = output_dir / "rq2_meaningful_interactions.csv"
    write_csv(interaction_csv, meaningful)

    stability_output = stability.sort_values(
        ["first_voxel_size", "second_voxel_size"], ignore_index=True
    )
    if len(stability_output) != 3:
        raise ValueError("expected three pairwise rank correlations")
    stability_csv = output_dir / "rq2_ranking_stability.csv"
    write_csv(stability_csv, stability_output)

    size_rows = []
    for row in size_output.itertuples(index=False):
        equivalent = bool_value(row.practically_equivalent_to_size_leader)
        size_rows.append(
            "{:.2f} & {:.4f} [{:.4f}, {:.4f}] & {:+.4f} [{:+.4f}, {:+.4f}] & {} \\\\".format(
                row.voxel_size,
                row.macro_roc_auc,
                row.roc_auc_ci_lower,
                row.roc_auc_ci_upper,
                row.delta_auc_vs_primary_0p02,
                row.delta_vs_primary_ci_lower,
                row.delta_vs_primary_ci_upper,
                "Yes" if equivalent else "No",
            )
        )
    interaction_rows = []
    for row in meaningful.itertuples(index=False):
        interaction_rows.append(
            "{:.2f} & {} & {:+.4f} & {:+.4f} [{:+.4f}, {:+.4f}] & {} & {:+.3f} \\\\".format(
                row.voxel_size,
                latex_escape(str(row.technique)),
                row.delta_macro_roc_auc,
                row.difference_in_differences_roc_auc,
                row.interaction_ci_lower,
                row.interaction_ci_upper,
                p_value_text(row.holm_adjusted_p_value),
                row.rank_biserial_interaction,
            )
        )
    stability_rows = [
        "{:.2f}--{:.2f} & {:.3f} \\\\".format(
            row.first_voxel_size,
            row.second_voxel_size,
            row.spearman_rank_correlation,
        )
        for row in stability_output.itertuples(index=False)
    ]
    latex = "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{Global RQ2 statistical analysis over the 16 matched "
            r"VLA--trajectory-condition blocks. Panel (a) reports the frozen "
            r"Harmonic Mean size comparison (Friedman $\chi^2_F(" +
            str(int(size_test["degrees_of_freedom"])) + ")=" +
            "{:.3f}".format(float(size_test["statistic"])) +
            r"$, $p=" + "{:.3f}".format(float(size_test["p_value"])) +
            r"$). Panel (b) contains interactions satisfying Holm-corrected "
            r"$p<0.05$ and $|\Delta\Delta\mathrm{AUC}|\geq0.01$. Panel (c) "
            r"reports tie-aware Spearman rank correlations across all 34 techniques.}",
            r"\label{tab:rq2-global-statistics}",
            r"{\scriptsize",
            r"\setlength{\tabcolsep}{4pt}",
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"\multicolumn{4}{l}{\textit{(a) Voxel-size effect for Harmonic Mean}} \\",
            r"Size & ROC-AUC [95\% CI] & $\Delta$ vs. 0.02 [95\% CI] & Equivalent \\",
            r"\midrule",
            *size_rows,
            r"\bottomrule",
            r"\end{tabular}",
            "",
            r"\vspace{2mm}",
            r"\begin{tabular}{llcccc}",
            r"\toprule",
            r"\multicolumn{6}{l}{\textit{(b) Practically meaningful technique--size interactions}} \\",
            r"Size vs. 0.02 & Technique & $\Delta$AUC & $\Delta\Delta$AUC [95\% CI] & Holm $p$ & $r_{rb}$ \\",
            r"\midrule",
            *interaction_rows,
            r"\bottomrule",
            r"\end{tabular}",
            "",
            r"\vspace{2mm}",
            r"\begin{tabular}{lc}",
            r"\toprule",
            r"\multicolumn{2}{l}{\textit{(c) Ranking stability}} \\",
            r"Voxel-size pair & Spearman $\rho$ \\",
            r"\midrule",
            *stability_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table*}",
            "",
        ]
    )
    tex_path = output_dir / "rq2_global_statistics.tex"
    atomic_text(tex_path, latex)
    return [size_csv, interaction_csv, stability_csv, tex_path], len(meaningful)


def build_supplementary_tables(
    vla_metrics: pd.DataFrame,
    output_dir: Path,
    cutoff: int,
) -> List[Path]:
    outputs: List[Path] = []
    for vla in VLA_ORDER:
        group = vla_metrics[vla_metrics["vla"] == vla]
        formula_ids = set(group["formula_id"].astype(str))
        if len(formula_ids) != EXPECTED_TECHNIQUES:
            raise ValueError("{} does not contain all 34 techniques".format(vla))
        rows = ordered_technique_rows(group, formula_ids)
        output = pd.DataFrame(rows)
        output.insert(0, "vla", VLA_DISPLAY[vla])
        csv_path = output_dir / "rq2_supplementary_{}.csv".format(VLA_DISPLAY[vla])
        write_csv(csv_path, output)
        outputs.append(csv_path)

        latex = "\n".join(
            [
                r"\begin{table*}[t]",
                r"\centering",
                r"\caption{Complete RQ2 technique ranking for " + latex_escape(VLA_DISPLAY[vla]) +
                r". Each cell reports macro ROC-AUC and the within-size rank in "
                r"parentheses. Ranks use unrounded values.}",
                r"\label{tab:rq2-supplementary-" + VLA_DISPLAY[vla].lower() + "}",
                r"{\scriptsize",
                r"\setlength{\tabcolsep}{7pt}",
                r"\renewcommand{\arraystretch}{0.96}",
                r"\begin{tabular}{lccc}",
                r"\toprule",
                r"Technique & 0.01 ROC-AUC (rank) & 0.02 ROC-AUC (rank) & 0.05 ROC-AUC (rank) \\",
                r"\midrule",
                *latex_technique_rows(rows, cutoff),
                r"\bottomrule",
                r"\end{tabular}",
                r"}",
                r"\end{table*}",
                "",
            ]
        )
        tex_path = output_dir / "rq2_supplementary_{}.tex".format(VLA_DISPLAY[vla])
        atomic_text(tex_path, latex)
        outputs.append(tex_path)
    return outputs


def generate_tables(args: argparse.Namespace) -> List[Path]:
    if args.top_rank_cutoff < 1:
        raise ValueError("top-rank-cutoff must be positive")
    aggregate_dir = args.aggregate_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    primary = read_csv(
        aggregate_dir / "primary_block_metrics.csv",
        (
            "voxel_size",
            "vla",
            "condition",
            "formula_id",
            "technique",
            "roc_auc",
            "average_precision",
            "mean_known_fraction",
            "score_tie_fraction",
        ),
    )
    folds = read_csv(
        aggregate_dir / "fold_metrics_by_size.csv.gz",
        (
            "voxel_size",
            "configuration_id",
            "vla",
            "condition",
            "budget",
            "test_run",
            "formula_id",
            "technique",
            "mean_known_fraction",
            "all_unknown_trajectories",
            "score_tie_fraction",
            "fit_seconds",
            "score_seconds",
            "model_artifact_bytes",
        ),
    )
    rankings = read_csv(
        aggregate_dir / "rankings_by_size.csv",
        ("voxel_size", "formula_id", "technique", "macro_roc_auc", "within_size_rank"),
    )
    global_diagnostics = read_csv(
        aggregate_dir / "size_diagnostics.csv",
        (
            "voxel_size",
            "mean_known_fraction",
            "all_unknown_trajectories",
            "mean_score_tie_fraction",
            "mean_fit_seconds",
            "mean_score_seconds",
            "mean_model_artifact_bytes",
            "folds",
            "mean_training_voxels",
        ),
    )
    size_comparison = read_csv(
        aggregate_dir / "voxel_size_comparison.csv",
        (
            "voxel_size",
            "macro_roc_auc",
            "roc_auc_ci_lower",
            "roc_auc_ci_upper",
            "delta_auc_vs_primary_0p02",
            "delta_vs_primary_ci_lower",
            "delta_vs_primary_ci_upper",
            "practically_equivalent_to_size_leader",
            "phase_a_size_rank",
        ),
    )
    interactions = read_csv(
        aggregate_dir / "technique_size_interactions.csv",
        (
            "voxel_size",
            "formula_id",
            "technique",
            "difference_in_differences_roc_auc",
            "interaction_ci_lower",
            "interaction_ci_upper",
            "holm_adjusted_p_value",
            "rank_biserial_interaction",
            "practically_meaningful_interaction",
        ),
    )
    effects = read_csv(
        aggregate_dir / "technique_size_effects.csv",
        (
            "voxel_size",
            "formula_id",
            "delta_macro_roc_auc",
            "delta_ci_lower",
            "delta_ci_upper",
            "holm_adjusted_p_value",
            "practically_meaningful_change",
        ),
    )
    stability = read_csv(
        aggregate_dir / "ranking_stability.csv",
        ("first_voxel_size", "second_voxel_size", "spearman_rank_correlation", "techniques"),
    )
    tests = read_json(aggregate_dir / "statistical_tests.json")

    validate_experiment_dimensions(primary, folds, rankings)
    vla_metrics = aggregate_vla_metrics(primary, rankings)
    sources = {
        0.01: args.fine_results,
        0.02: args.primary_results,
        0.05: args.coarse_results,
    }
    training_voxels = load_training_voxels(sources)

    outputs: List[Path] = []
    outputs.extend(build_reference_table(vla_metrics, rankings, output_dir))
    competitive_csv, competitive_tex, included = build_competitive_technique_table(
        vla_metrics, output_dir, args.top_rank_cutoff
    )
    outputs.extend((competitive_csv, competitive_tex))
    outputs.extend(
        build_diagnostics_table(folds, training_voxels, global_diagnostics, output_dir)
    )
    statistical_outputs, meaningful_count = build_statistical_tables(
        size_comparison,
        interactions,
        effects,
        stability,
        tests,
        output_dir,
    )
    outputs.extend(statistical_outputs)
    outputs.extend(
        build_supplementary_tables(vla_metrics, output_dir, args.top_rank_cutoff)
    )

    manifest = {
        "source_directory": relative_or_absolute(aggregate_dir),
        "source_sha256": {
            filename: sha256_file(aggregate_dir / filename)
            for filename in AGGREGATE_INPUTS
        },
        "training_voxel_sources": {
            SIZE_LABEL[size]: relative_or_absolute(Path(source))
            for size, source in sources.items()
        },
        "voxel_sizes": list(VOXEL_SIZES),
        "vla_order": [VLA_DISPLAY[vla] for vla in VLA_ORDER],
        "technique_count": EXPECTED_TECHNIQUES,
        "top_rank_cutoff": args.top_rank_cutoff,
        "competitive_formula_ids_by_vla": {
            VLA_DISPLAY[vla]: formula_ids for vla, formula_ids in included.items()
        },
        "meaningful_interactions": meaningful_count,
        "statistical_scope": "global over 16 matched VLA-condition blocks",
        "descriptive_scope": "stratified by VLA",
        "outputs": [path.name for path in outputs],
    }
    manifest_path = output_dir / "manifest.json"
    atomic_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    outputs.append(manifest_path)
    return outputs


def main() -> None:
    args = parse_args()
    outputs = generate_tables(args)
    print("Generated {} RQ2 table artifacts in {}".format(len(outputs), args.output_dir.resolve()))
    for path in outputs:
        print("- {}".format(path.resolve()))


if __name__ == "__main__":
    main()
