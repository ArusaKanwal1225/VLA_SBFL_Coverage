#!/usr/bin/env python3
"""Generate the main-paper and supplementary RQ3 v2 tables.

The experiment runner owns all aggregation and statistical testing.  This
script only validates those frozen outputs and reformats them as auditable CSV
files and LaTeX tables; it never recomputes a p-value from raw observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


VOXEL_SIZE = 0.02
REFERENCE_FORMULA = "harmonic_mean"
EXPECTED_TECHNIQUES = 34
TRAINING_SIZES = (10, 20, 50, 100, 200)
VLA_ORDER = ("EO1", "Groot", "Pi0", "SpatialVLA")
VLA_DISPLAY = {
    "EO1": "EO1",
    "Groot": "GR00T",
    "Pi0": "Pi0",
    "SpatialVLA": "SpatialVLA",
}
VLA_FIELD = {
    "EO1": "eo1",
    "Groot": "gr00t",
    "Pi0": "pi0",
    "SpatialVLA": "spatialvla",
}
INPUT_FILES = (
    "adjacent_size_tests.csv",
    "conditional_coverage_tests.csv",
    "global_sample_efficiency.csv",
    "learning_curves_by_vla.csv",
    "sample_efficiency_by_vla.csv",
)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    result_root = (
        repository_root()
        / "experiments"
        / "RQ3_v2"
        / "results"
        / "learning_curves"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aggregate-dir", type=Path, default=result_root / "aggregate"
    )
    parser.add_argument("--output-dir", type=Path, default=result_root / "tables")
    parser.add_argument("--statistical-review-method", choices=("approximate", "exact"))
    parser.add_argument(
        "--top-rank-cutoff",
        type=int,
        default=5,
        help=(
            "Include a technique in the main sample-efficiency table when it "
            "reaches this NAULC rank for at least one VLA."
        ),
    )
    return parser.parse_args()


def read_csv(path: Path, required: Iterable[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(
            "{} is missing columns: {}".format(path, ", ".join(sorted(missing)))
        )
    if frame.empty:
        raise ValueError("{} contains no rows".format(path))
    return frame


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


def p_value_text(value: float) -> str:
    if value < 0.001:
        return r"$<0.001$"
    return "{:.4f}".format(value)


def rank_text(value: float) -> str:
    rounded = round(float(value))
    return str(int(rounded)) if math.isclose(float(value), rounded) else "{:.1f}".format(value)


def validate_dimensions(
    curves: pd.DataFrame,
    efficiency_by_vla: pd.DataFrame,
    global_efficiency: pd.DataFrame,
    adjacent: pd.DataFrame,
    conditional: pd.DataFrame,
) -> None:
    frames = {
        "learning curves": curves,
        "VLA sample efficiency": efficiency_by_vla,
        "global sample efficiency": global_efficiency,
    }
    for name, frame in frames.items():
        sizes = {float(value) for value in frame["voxel_size"].unique()}
        if sizes != {VOXEL_SIZE}:
            raise ValueError("{} must contain only voxel size 0.02".format(name))

    expected_vlas = set(VLA_ORDER)
    if set(curves["vla"].astype(str)) != expected_vlas:
        raise ValueError("learning curves do not contain exactly the four expected VLAs")
    if set(efficiency_by_vla["vla"].astype(str)) != expected_vlas:
        raise ValueError("sample efficiency does not contain exactly the four expected VLAs")
    if set(curves["training_size"].astype(int)) != set(TRAINING_SIZES):
        raise ValueError("learning curves do not contain exactly the five expected sizes")

    curve_counts = curves.groupby(["vla", "training_size"])["formula_id"].nunique()
    if len(curve_counts) != len(VLA_ORDER) * len(TRAINING_SIZES):
        raise ValueError("the VLA-size learning-curve grid is incomplete")
    if set(curve_counts.astype(int)) != {EXPECTED_TECHNIQUES}:
        raise ValueError("every VLA-size cell must contain 34 techniques")
    efficiency_counts = efficiency_by_vla.groupby("vla")["formula_id"].nunique()
    if set(efficiency_counts.astype(int)) != {EXPECTED_TECHNIQUES}:
        raise ValueError("every VLA must contain 34 sample-efficiency summaries")
    if global_efficiency["formula_id"].nunique() != EXPECTED_TECHNIQUES:
        raise ValueError("global sample efficiency must contain 34 techniques")
    if conditional["formula_id"].nunique() != EXPECTED_TECHNIQUES:
        raise ValueError("conditional coverage tests must contain 34 techniques")

    expected_pairs = set(zip(TRAINING_SIZES[:-1], TRAINING_SIZES[1:]))
    observed_pairs = set(
        zip(
            adjacent["lower_training_size"].astype(int),
            adjacent["upper_training_size"].astype(int),
        )
    )
    if observed_pairs != expected_pairs or len(adjacent) != len(expected_pairs):
        raise ValueError("adjacent-size tests do not contain the four planned contrasts")
    if set(adjacent["primary_blocks"].astype(int)) != {16}:
        raise ValueError("adjacent-size tests must use 16 matched primary blocks")
    if set(conditional["primary_blocks"].astype(int)) != {16}:
        raise ValueError("conditional coverage tests must use 16 primary blocks")

    label_frames = [
        curves[["formula_id", "technique"]],
        efficiency_by_vla[["formula_id", "technique"]],
        global_efficiency[["formula_id", "technique"]],
        conditional[["formula_id", "technique"]],
    ]
    labels = pd.concat(label_frames, ignore_index=True).drop_duplicates()
    if labels.groupby("formula_id")["technique"].nunique().max() != 1:
        raise ValueError("a formula identifier maps to multiple technique names")

    numeric_columns = {
        "learning curves": [
            "roc_auc",
            "calibration_unique_coverage",
            "test_unique_coverage",
            "roc_auc_rank",
        ],
        "VLA sample efficiency": [
            "normalized_log_aulc",
            "auc_at_minimum_size",
            "auc_at_maximum_size",
            "gain_minimum_to_maximum",
            "naulc_rank",
        ],
        "global sample efficiency": [
            "normalized_log_aulc",
            "auc_at_maximum_size",
            "gain_minimum_to_maximum",
            "naulc_rank",
        ],
        "adjacent tests": [
            "mean_delta_roc_auc",
            "median_delta_roc_auc",
            "positive_block_fraction",
            "holm_adjusted_p_value",
            "rank_biserial",
        ],
        "conditional tests": [
            "mean_roc_auc_per_10pp_coverage",
            "holm_adjusted_p_value",
        ],
    }
    numeric_frames = {
        **frames,
        "adjacent tests": adjacent,
        "conditional tests": conditional,
    }
    for name, columns in numeric_columns.items():
        values = numeric_frames[name][columns].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("{} contains non-finite publication values".format(name))


def build_coverage_performance_data(curves: pd.DataFrame) -> pd.DataFrame:
    """Build one row per VLA and suite size for the first main table."""

    rows: list[dict[str, object]] = []
    for vla in VLA_ORDER:
        for size in TRAINING_SIZES:
            group = curves[
                (curves["vla"].astype(str) == vla)
                & (curves["training_size"].astype(int) == size)
            ]
            reference = group[group["formula_id"].astype(str) == REFERENCE_FORMULA]
            if len(reference) != 1:
                raise ValueError(
                    "expected one Harmonic Mean row for {} at n={}".format(vla, size)
                )
            maximum = float(group["roc_auc"].max())
            leaders = group[np.isclose(group["roc_auc"].astype(float), maximum, atol=1e-12)]
            leaders = leaders.sort_values("technique", kind="mergesort")
            row = reference.iloc[0]
            rows.append(
                {
                    "vla": vla,
                    "vla_display": VLA_DISPLAY[vla],
                    "training_size": size,
                    "calibration_unique_coverage": float(
                        row["calibration_unique_coverage"]
                    ),
                    "test_unique_coverage": float(row["test_unique_coverage"]),
                    "harmonic_mean_roc_auc": float(row["roc_auc"]),
                    "harmonic_mean_rank": float(row["roc_auc_rank"]),
                    "leading_formula_ids": " / ".join(
                        leaders["formula_id"].astype(str)
                    ),
                    "leading_techniques": " / ".join(
                        leaders["technique"].astype(str)
                    ),
                    "leading_roc_auc": maximum,
                }
            )
    return pd.DataFrame(rows)


def competitive_formula_ids(
    efficiency_by_vla: pd.DataFrame, cutoff: int
) -> tuple[str, ...]:
    if cutoff < 1:
        raise ValueError("top-rank cutoff must be positive")
    selected = set(
        efficiency_by_vla.loc[
            efficiency_by_vla["naulc_rank"].astype(float) <= cutoff, "formula_id"
        ].astype(str)
    )
    selected.add(REFERENCE_FORMULA)
    order = (
        efficiency_by_vla.groupby("formula_id", as_index=False)[
            "normalized_log_aulc"
        ]
        .mean()
        .sort_values(
            ["normalized_log_aulc", "formula_id"],
            ascending=[False, True],
            kind="mergesort",
        )
    )
    return tuple(
        order.loc[order["formula_id"].astype(str).isin(selected), "formula_id"].astype(str)
    )


def build_sample_efficiency_data(
    efficiency_by_vla: pd.DataFrame,
    global_efficiency: pd.DataFrame,
    formula_ids: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for formula_id in formula_ids:
        global_rows = global_efficiency[
            global_efficiency["formula_id"].astype(str) == formula_id
        ]
        if len(global_rows) != 1:
            raise ValueError("missing global sample-efficiency row for " + formula_id)
        global_row = global_rows.iloc[0]
        row: dict[str, object] = {
            "formula_id": formula_id,
            "technique": str(global_row["technique"]),
        }
        for vla in VLA_ORDER:
            selected = efficiency_by_vla[
                (efficiency_by_vla["vla"].astype(str) == vla)
                & (efficiency_by_vla["formula_id"].astype(str) == formula_id)
            ]
            if len(selected) != 1:
                raise ValueError(
                    "missing sample-efficiency row for {} and {}".format(
                        formula_id, vla
                    )
                )
            value = selected.iloc[0]
            field = VLA_FIELD[vla]
            row[field + "_naulc"] = float(value["normalized_log_aulc"])
            row[field + "_rank"] = float(value["naulc_rank"])
        row.update(
            {
                "macro_naulc": float(global_row["normalized_log_aulc"]),
                "macro_rank": float(global_row["naulc_rank"]),
                "macro_auc_at_200": float(global_row["auc_at_maximum_size"]),
                "macro_gain_10_to_200": float(
                    global_row["gain_minimum_to_maximum"]
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["macro_rank", "formula_id"], kind="mergesort", ignore_index=True
    )


def build_main_coverage_table(data: pd.DataFrame, output_dir: Path) -> list[Path]:
    csv_path = output_dir / "rq3v2_main_coverage_performance_by_vla.csv"
    write_csv(csv_path, data)
    latex_rows: list[str] = []
    for vla in VLA_ORDER:
        group = data[data["vla"].astype(str) == vla]
        for index, row in enumerate(group.itertuples(index=False)):
            latex_rows.append(
                "{} & {} & {:.1f} & {:.1f} & {:.4f} ({}) & {} & {:.4f} \\\\".format(
                    latex_escape(VLA_DISPLAY[vla]) if index == 0 else "",
                    int(row.training_size),
                    100.0 * row.calibration_unique_coverage,
                    100.0 * row.test_unique_coverage,
                    row.harmonic_mean_roc_auc,
                    rank_text(row.harmonic_mean_rank),
                    latex_escape(str(row.leading_techniques)),
                    row.leading_roc_auc,
                )
            )
        if vla != VLA_ORDER[-1]:
            latex_rows.append(r"\midrule")
    latex = "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{RQ3 learning-curve coverage and held-out performance by "
            r"VLA. Coverage values are percentages. The Harmonic Mean cell "
            r"reports ROC-AUC and its within-VLA, within-size rank in "
            r"parentheses; leaders were determined from unrounded ROC-AUC.}",
            r"\label{tab:rq3v2-coverage-performance}",
            r"{\scriptsize",
            r"\setlength{\tabcolsep}{4pt}",
            r"\begin{tabular}{lrrrrlr}",
            r"\toprule",
            r"VLA & $n$ & Calibration cov. (\%) & Held-out cov. (\%) & "
            "Harmonic Mean AUC (rank) & Leading technique & Leading AUC \\\\",
            r"\midrule",
            *latex_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table*}",
            "",
        ]
    )
    tex_path = output_dir / "rq3v2_main_coverage_performance_by_vla.tex"
    atomic_text(tex_path, latex)
    return [csv_path, tex_path]


def naulc_rank_cell(naulc: float, rank: float) -> str:
    content = "{:.4f} ({})".format(naulc, rank_text(rank))
    if math.isclose(float(rank), 1.0):
        return r"\textbf{" + content + "}"
    if float(rank) <= 5.0:
        return r"\underline{" + content + "}"
    return content


def build_main_efficiency_table(data: pd.DataFrame, output_dir: Path) -> list[Path]:
    csv_path = output_dir / "rq3v2_main_sample_efficiency_by_vla.csv"
    write_csv(csv_path, data)
    rows = []
    for row in data.itertuples(index=False):
        cells = [latex_escape(str(row.technique))]
        for vla in VLA_ORDER:
            field = VLA_FIELD[vla]
            cells.append(
                naulc_rank_cell(
                    float(getattr(row, field + "_naulc")),
                    float(getattr(row, field + "_rank")),
                )
            )
        cells.extend(
            [
                naulc_rank_cell(row.macro_naulc, row.macro_rank),
                "{:.4f}".format(row.macro_auc_at_200),
                "{:+.4f}".format(row.macro_gain_10_to_200),
            ]
        )
        rows.append(" & ".join(cells) + r" \\")
    latex = "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{RQ3 sample efficiency for techniques reaching a top-five "
            r"NAULC rank for at least one VLA. VLA and macro cells report NAULC "
            r"with rank in parentheses. Bold denotes rank 1 and underlining the "
            r"remaining top-five ranks. Endpoint and gain are macro ROC-AUC.}",
            r"\label{tab:rq3v2-sample-efficiency}",
            r"{\scriptsize",
            r"\setlength{\tabcolsep}{3.5pt}",
            r"\begin{tabular}{lcccccrr}",
            r"\toprule",
            r"Technique & EO1 & GR00T & Pi0 & SpatialVLA & Macro & AUC@200 & "
            "$\\Delta$AUC$_{10\\rightarrow200}$ \\\\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table*}",
            "",
        ]
    )
    tex_path = output_dir / "rq3v2_main_sample_efficiency_by_vla.tex"
    atomic_text(tex_path, latex)
    return [csv_path, tex_path]


def build_main_adjacent_table(adjacent: pd.DataFrame, output_dir: Path) -> list[Path]:
    ordered = adjacent.sort_values("lower_training_size", kind="mergesort").copy()
    csv_path = output_dir / "rq3v2_main_adjacent_size_effects.csv"
    write_csv(csv_path, ordered)
    rows = [
        "{}--{} & {:+.4f} & {:+.4f} & {:.1f} & {} & {:+.3f} \\\\".format(
            int(row.lower_training_size),
            int(row.upper_training_size),
            row.mean_delta_roc_auc,
            row.median_delta_roc_auc,
            100.0 * row.positive_block_fraction,
            p_value_text(row.holm_adjusted_p_value),
            row.rank_biserial,
        )
        for row in ordered.itertuples(index=False)
    ]
    latex = "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{RQ3 matched adjacent-size effects for Harmonic Mean over "
            r"the 16 VLA--trajectory-generation blocks. Tests used paired "
            r"Wilcoxon signed-rank tests with one Holm correction.}",
            r"\label{tab:rq3v2-adjacent-effects}",
            r"{\scriptsize",
            r"\setlength{\tabcolsep}{4pt}",
            r"\begin{tabular}{lrrrrr}",
            r"\toprule",
            r"Transition & Mean $\Delta$AUC & Median $\Delta$AUC & Positive (\%) "
            "& Holm $p$ & $r_{rb}$ \\\\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table}",
            "",
        ]
    )
    tex_path = output_dir / "rq3v2_main_adjacent_size_effects.tex"
    atomic_text(tex_path, latex)
    return [csv_path, tex_path]


def learning_curve_rows(group: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    records = []
    latex_rows = []
    formula_order = (
        group.groupby(["formula_id", "technique"], as_index=False)["roc_auc"]
        .mean()
        .sort_values(
            ["roc_auc", "formula_id"],
            ascending=[False, True],
            kind="mergesort",
        )
    )
    for summary in formula_order.itertuples(index=False):
        selected = group[group["formula_id"].astype(str) == str(summary.formula_id)]
        record: dict[str, object] = {
            "formula_id": str(summary.formula_id),
            "technique": str(summary.technique),
        }
        cells = [latex_escape(str(summary.technique))]
        for size in TRAINING_SIZES:
            row = selected[selected["training_size"].astype(int) == size]
            if len(row) != 1:
                raise ValueError(
                    "incomplete learning curve for {} at n={}".format(
                        summary.formula_id, size
                    )
                )
            item = row.iloc[0]
            auc = float(item["roc_auc"])
            rank = float(item["roc_auc_rank"])
            record["auc_at_{}".format(size)] = auc
            record["rank_at_{}".format(size)] = rank
            cells.append(naulc_rank_cell(auc, rank))
        records.append(record)
        latex_rows.append(" & ".join(cells) + r" \\")
    return pd.DataFrame(records), latex_rows


def build_supplementary_learning_curves(
    curves: pd.DataFrame, output_dir: Path
) -> list[Path]:
    outputs: list[Path] = []
    combined: list[pd.DataFrame] = []
    for vla in VLA_ORDER:
        data, rows = learning_curve_rows(curves[curves["vla"].astype(str) == vla])
        data.insert(0, "vla", vla)
        data.insert(1, "vla_display", VLA_DISPLAY[vla])
        combined.append(data)
        latex = "\n".join(
            [
                r"\begin{table*}[t]",
                r"\centering",
                r"\caption{Complete RQ3 ROC-AUC learning curves for "
                + latex_escape(VLA_DISPLAY[vla])
                + r". Each cell reports ROC-AUC and the within-size rank in "
                r"parentheses. Bold denotes rank 1 and underlining the remaining "
                r"top-five ranks.}",
                r"\label{tab:rq3v2-learning-curves-" + VLA_FIELD[vla] + "}",
                r"{\scriptsize",
                r"\setlength{\tabcolsep}{5pt}",
                r"\begin{tabular}{lccccc}",
                r"\toprule",
                "Technique & $n=10$ & $n=20$ & $n=50$ & $n=100$ & $n=200$ \\\\",
                r"\midrule",
                *rows,
                r"\bottomrule",
                r"\end{tabular}",
                r"}",
                r"\end{table*}",
                "",
            ]
        )
        path = output_dir / (
            "rq3v2_supplementary_learning_curves_" + VLA_FIELD[vla] + ".tex"
        )
        atomic_text(path, latex)
        outputs.append(path)
    csv_path = output_dir / "rq3v2_supplementary_learning_curves_all.csv"
    write_csv(csv_path, pd.concat(combined, ignore_index=True))
    return [csv_path, *outputs]


def build_supplementary_efficiency(
    efficiency_by_vla: pd.DataFrame, output_dir: Path
) -> list[Path]:
    outputs: list[Path] = []
    ordered_frames = []
    for vla in VLA_ORDER:
        group = efficiency_by_vla[
            efficiency_by_vla["vla"].astype(str) == vla
        ].sort_values(["naulc_rank", "formula_id"], kind="mergesort")
        ordered_frames.append(group)
        rows = []
        for row in group.itertuples(index=False):
            rows.append(
                "{} & {:.4f} & {} & {:.4f} & {:.4f} & {:+.4f} & {:.1f} \\\\".format(
                    latex_escape(str(row.technique)),
                    row.normalized_log_aulc,
                    rank_text(row.naulc_rank),
                    row.auc_at_minimum_size,
                    row.auc_at_maximum_size,
                    row.gain_minimum_to_maximum,
                    row.size_at_95_percent_improvement,
                )
            )
        latex = "\n".join(
            [
                r"\begin{table*}[t]",
                r"\centering",
                r"\caption{Complete RQ3 sample-efficiency results for "
                + latex_escape(VLA_DISPLAY[vla])
                + r". NAULC ranks use unrounded values; $n_{95}$ is descriptive.}",
                r"\label{tab:rq3v2-efficiency-" + VLA_FIELD[vla] + "}",
                r"{\scriptsize",
                r"\setlength{\tabcolsep}{6pt}",
                r"\begin{tabular}{lrrrrrr}",
                r"\toprule",
                "Technique & NAULC & Rank & AUC@10 & AUC@200 & Gain & $n_{95}$ \\\\",
                r"\midrule",
                *rows,
                r"\bottomrule",
                r"\end{tabular}",
                r"}",
                r"\end{table*}",
                "",
            ]
        )
        path = output_dir / (
            "rq3v2_supplementary_sample_efficiency_" + VLA_FIELD[vla] + ".tex"
        )
        atomic_text(path, latex)
        outputs.append(path)
    csv_path = output_dir / "rq3v2_supplementary_sample_efficiency_all.csv"
    write_csv(csv_path, pd.concat(ordered_frames, ignore_index=True))
    return [csv_path, *outputs]


def build_supplementary_conditional(
    conditional: pd.DataFrame, output_dir: Path
) -> list[Path]:
    ordered = conditional.sort_values(
        ["mean_roc_auc_per_10pp_coverage", "formula_id"],
        ascending=[False, True],
        kind="mergesort",
    ).copy()
    csv_path = output_dir / "rq3v2_supplementary_conditional_coverage.csv"
    write_csv(csv_path, ordered)
    rows = [
        "{} & {:+.4f} & {:+.4f} & {:.1f} & {} & {:+.3f} \\\\".format(
            latex_escape(str(row.technique)),
            row.mean_roc_auc_per_10pp_coverage,
            row.median_roc_auc_per_10pp_coverage,
            100.0 * row.positive_block_fraction,
            p_value_text(row.holm_adjusted_p_value),
            row.rank_biserial,
        )
        for row in ordered.itertuples(index=False)
    ]
    latex = "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{Complete conditional coverage analysis for RQ3. Effects "
            r"are ROC-AUC changes per ten percentage points of calibration "
            r"coverage after centering within configuration, outer fold, and "
            r"suite size. Holm correction covered all 34 techniques.}",
            r"\label{tab:rq3v2-conditional-coverage}",
            r"{\scriptsize",
            r"\setlength{\tabcolsep}{6pt}",
            r"\begin{tabular}{lrrrrr}",
            r"\toprule",
            r"Technique & Mean effect & Median effect & Positive blocks (\%) & "
            "Holm $p$ & $r_{rb}$ \\\\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table*}",
            "",
        ]
    )
    tex_path = output_dir / "rq3v2_supplementary_conditional_coverage.tex"
    atomic_text(tex_path, latex)
    return [csv_path, tex_path]


def build_index(outputs: Sequence[Path], output_dir: Path) -> Path:
    main = sorted(path.name for path in outputs if "_main_" in path.name)
    supplementary = sorted(
        path.name for path in outputs if "_supplementary_" in path.name
    )
    markdown = "\n".join(
        [
            "# RQ3 v2 publication tables",
            "",
            "Generated from the completed learning-curve aggregate outputs.",
            "",
            "## Main paper",
            "",
            *["- `{}`".format(name) for name in main],
            "",
            "## Supplementary material",
            "",
            *["- `{}`".format(name) for name in supplementary],
            "",
            "LaTeX tables require `booktabs`. Percentages are rendered as percentages;",
            "their CSV counterparts retain proportions in `[0,1]`.",
            "",
        ]
    )
    path = output_dir / "TABLES.md"
    atomic_text(path, markdown)
    return path


def build_manifest(
    aggregate_dir: Path,
    output_dir: Path,
    outputs: Sequence[Path],
    cutoff: int,
) -> Path:
    manifest = {
        "voxel_size": VOXEL_SIZE,
        "training_sizes": list(TRAINING_SIZES),
        "vlas": list(VLA_ORDER),
        "top_rank_cutoff": cutoff,
        "inputs": {
            name: sha256_file(aggregate_dir / name) for name in INPUT_FILES
        },
        "outputs": sorted(path.name for path in outputs),
    }
    path = output_dir / "manifest.json"
    atomic_text(path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def generate_tables(args: argparse.Namespace) -> list[Path]:
    aggregate_dir = args.aggregate_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    curves = read_csv(
        aggregate_dir / "learning_curves_by_vla.csv",
        (
            "voxel_size",
            "vla",
            "training_size",
            "formula_id",
            "technique",
            "roc_auc",
            "calibration_unique_coverage",
            "test_unique_coverage",
            "roc_auc_rank",
        ),
    )
    efficiency_by_vla = read_csv(
        aggregate_dir / "sample_efficiency_by_vla.csv",
        (
            "voxel_size",
            "vla",
            "formula_id",
            "technique",
            "normalized_log_aulc",
            "auc_at_minimum_size",
            "auc_at_maximum_size",
            "gain_minimum_to_maximum",
            "size_at_95_percent_improvement",
            "naulc_rank",
        ),
    )
    global_efficiency = read_csv(
        aggregate_dir / "global_sample_efficiency.csv",
        (
            "voxel_size",
            "formula_id",
            "technique",
            "normalized_log_aulc",
            "auc_at_maximum_size",
            "gain_minimum_to_maximum",
            "naulc_rank",
        ),
    )
    adjacent = read_csv(
        aggregate_dir / "adjacent_size_tests.csv",
        (
            "lower_training_size",
            "upper_training_size",
            "mean_delta_roc_auc",
            "median_delta_roc_auc",
            "positive_block_fraction",
            "holm_adjusted_p_value",
            "rank_biserial",
            "primary_blocks",
        ),
    )
    conditional = read_csv(
        aggregate_dir / "conditional_coverage_tests.csv",
        (
            "formula_id",
            "technique",
            "mean_roc_auc_per_10pp_coverage",
            "median_roc_auc_per_10pp_coverage",
            "positive_block_fraction",
            "holm_adjusted_p_value",
            "rank_biserial",
            "primary_blocks",
        ),
    )
    validate_dimensions(
        curves, efficiency_by_vla, global_efficiency, adjacent, conditional
    )

    coverage = build_coverage_performance_data(curves)
    formula_ids = competitive_formula_ids(
        efficiency_by_vla, args.top_rank_cutoff
    )
    efficiency = build_sample_efficiency_data(
        efficiency_by_vla, global_efficiency, formula_ids
    )
    outputs: list[Path] = []
    outputs.extend(build_main_coverage_table(coverage, output_dir))
    outputs.extend(build_main_efficiency_table(efficiency, output_dir))
    outputs.extend(build_main_adjacent_table(adjacent, output_dir))
    outputs.extend(build_supplementary_learning_curves(curves, output_dir))
    outputs.extend(build_supplementary_efficiency(efficiency_by_vla, output_dir))
    outputs.extend(build_supplementary_conditional(conditional, output_dir))
    outputs.append(build_index(outputs, output_dir))
    outputs.append(
        build_manifest(
            aggregate_dir,
            output_dir,
            outputs,
            args.top_rank_cutoff,
        )
    )
    return outputs


def generate_statistical_review_tables(args: argparse.Namespace) -> list[Path]:
    """Render the three separate frozen families without loading experiment cells."""
    method = args.statistical_review_method
    aggregate = args.aggregate_dir.resolve()
    output = args.output_dir.resolve()
    specs = (
        ("adjacent_size_tests", 4, "mean_delta_roc_auc"),
        ("sample_efficiency_tests", 33, "difference_vs_harmonic_mean"),
        ("conditional_coverage_tests", 34, "mean_roc_auc_per_10pp_coverage"),
    )
    review = aggregate / f"statistical_review_{method}.json"
    metadata = json.loads(review.read_text())
    if metadata.get("wilcoxon_method") != method or metadata.get("families") != {
        "adjacent": 4, "sample_efficiency": 33, "conditional_coverage": 34}:
        raise ValueError("inconsistent RQ4 review metadata")
    # Validate every source before creating any table.
    for stem, family, effect in specs:
        frame = read_csv(aggregate / f"{stem}_{method}.csv", [effect, "wilcoxon_p_value",
            "holm_adjusted_p_value", "rank_biserial", "holm_family_size", "wilcoxon_method"])
        if len(frame) != (34 if family == 33 else family) or set(frame.holm_family_size) != {family} or set(frame.wilcoxon_method) != {method}:
            raise ValueError("invalid review family: " + stem)
    outputs, sources = [], {}
    for stem, family, effect in specs:
        source = aggregate / f"{stem}_{method}.csv"
        frame = read_csv(source, [effect, "wilcoxon_p_value", "holm_adjusted_p_value",
                                  "rank_biserial", "holm_family_size", "wilcoxon_method"])
        expected = 34 if family == 33 else family
        if len(frame) != expected or set(frame.holm_family_size) != {family} or set(frame.wilcoxon_method) != {method}:
            raise ValueError("invalid review family: " + stem)
        csv_path = output / f"rq3v2_{stem}_{method}.csv"
        write_csv(csv_path, frame)
        lines = [r"\begin{longtable}{lrrrr}",
                 r"\caption{RQ4 " + latex_escape(stem) + ": two-sided " + method
                 + " Wilcoxon signed-rank, Holm over " + str(family) + r" tests.}\\",
                 r"\toprule", r"Comparison & Mean effect & Raw $p$ & Holm $p$ & $r_{rb}$ \\", r"\midrule"]
        for row in frame.to_dict("records"):
            label = row.get("technique", str(row.get("lower_training_size", "")) + "--" + str(row.get("upper_training_size", "")))
            lines.append("{} & {:+.6f} & {:.8g} & {:.8g} & {:+.3f} {}".format(
                latex_escape(str(label)), row[effect], row["wilcoxon_p_value"],
                row["holm_adjusted_p_value"], row["rank_biserial"], r"\\"))
        lines.extend([r"\bottomrule", r"\end{longtable}", ""])
        tex_path = csv_path.with_suffix(".tex")
        atomic_text(tex_path, "\n".join(lines))
        outputs.extend([csv_path, tex_path])
        sources[source.name] = sha256_file(source)
    review = aggregate / f"statistical_review_{method}.json"
    sources[review.name] = sha256_file(review)
    manifest = output / f"statistical_review_{method}_manifest.json"
    atomic_text(manifest, json.dumps({"method": method, "source_sha256": sources,
        "outputs": {p.name: sha256_file(p) for p in outputs}}, indent=2) + "\n")
    return outputs + [manifest]


def main() -> int:
    args = parse_args()
    outputs = (generate_statistical_review_tables(args)
               if args.statistical_review_method else generate_tables(args))
    print("Wrote {} RQ3 v2 table artifacts to {}".format(len(outputs), args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
