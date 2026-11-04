#!/usr/bin/env python3
"""Generate CSV and LaTeX tables for the RQ2 baseline comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd


VLA_ORDER = ("EO1", "Groot", "Pi0", "SpatialVLA")
VLA_DISPLAY = {"EO1": "EO1", "Groot": "GR00T", "Pi0": "Pi0", "SpatialVLA": "SpatialVLA"}


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    root = repository_root()
    result_root = root / "experiments" / "RQ2_baselines" / "results" / "baseline_comparison"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-dir", type=Path, default=result_root / "aggregate")
    parser.add_argument("--output-dir", type=Path, default=result_root / "tables")
    parser.add_argument("--rq1-aggregate-dir", type=Path, default=root / "experiments" / "RQ1" / "results" / "phase_a_primary" / "aggregate")
    return parser.parse_args()


def read_csv(path: Path, required: Iterable[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError("{} is missing columns: {}".format(path, ", ".join(sorted(missing))))
    if frame.empty:
        raise ValueError(str(path) + " is empty")
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


def latex_escape(value: object) -> str:
    replacements = {
        "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
        "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}", "\\": r"\textbackslash{}",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def boolean(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def performance_table(
    by_vla: pd.DataFrame,
    overall: pd.DataFrame,
    output_dir: Path,
) -> Tuple[Path, Path]:
    complete = overall[overall["complete_primary_comparison"].map(boolean)].copy()
    available_vlas = [vla for vla in VLA_ORDER if vla in set(by_vla["vla"].astype(str))]
    if not available_vlas:
        raise ValueError("performance data contain no recognized VLA")
    all_vlas_complete = set(available_vlas) == set(VLA_ORDER)
    method_order = complete.sort_values(
        ["macro_roc_auc", "method_id"], ascending=[False, True]
    )["method_id"].tolist()
    rows: List[dict] = []
    for method_id in method_order:
        metadata = complete[complete["method_id"] == method_id].iloc[0]
        name = str(metadata["method"]) + ("$^{\\dagger}$" if boolean(metadata["task_aware"]) else "")
        row = {
            "method_id": method_id,
            "Method": str(metadata["method"]),
            "Family": str(metadata["family"]),
            "task_aware": boolean(metadata["task_aware"]),
        }
        for vla in VLA_ORDER:
            cell = by_vla[(by_vla["method_id"] == method_id) & (by_vla["vla"] == vla)]
            if vla in available_vlas and len(cell) != 1:
                raise ValueError("missing or duplicate VLA cell for " + method_id)
            row[VLA_DISPLAY[vla]] = (
                float(cell.iloc[0]["macro_roc_auc"]) if len(cell) == 1 else float("nan")
            )
        # An EO1-only macro-average is not an overall cross-VLA result. Keep the
        # final columns empty until the complete four-VLA design is available.
        row["Overall"] = (
            float(metadata["macro_roc_auc"]) if all_vlas_complete else float("nan")
        )
        row["Rank"] = (
            int(float(metadata["overall_rank"])) if all_vlas_complete else float("nan")
        )
        row["latex_name"] = latex_escape(str(metadata["method"])) + (
            r"$^{\dagger}$" if boolean(metadata["task_aware"]) else ""
        )
        rows.append(row)
    frame = pd.DataFrame(rows)
    csv_columns = ["Method", "Family", *[VLA_DISPLAY[vla] for vla in VLA_ORDER], "Overall", "Rank", "task_aware"]
    csv_path = output_dir / "rq2_baseline_performance.csv"
    write_csv(csv_path, frame[csv_columns])

    numeric_columns = [*[VLA_DISPLAY[vla] for vla in VLA_ORDER], "Overall"]
    best_by_column = {
        column: frame[column].max() if frame[column].notna().any() else float("nan")
        for column in numeric_columns
    }
    second_best_by_column = {}
    for column in numeric_columns:
        values = sorted(set(frame[column].dropna().astype(float)), reverse=True)
        second_best_by_column[column] = next(
            (value for value in values
             if abs(value - float(best_by_column[column])) >= 1e-12),
            float("nan"),
        )
    header = " & ".join(["Method", *[VLA_DISPLAY[vla] for vla in VLA_ORDER], "Overall", "Rank"])
    pending_note = (
        " Blank cells denote VLA results that have not yet been executed."
        if not all_vlas_complete else ""
    )
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{RQ2 -- Trajectory-level failure-discrimination ROC-AUC. Harmonic Mean is the frozen VoxelSBFL reference from RQ1. Bold denotes the best result and underlining denotes the second-best result in each populated column. $^{\dagger}$ denotes a task-aware metric using object/goal state."
        + pending_note + "}",
        r"\label{tab:rq2-baseline-performance}",
        r"{\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l" + "c" * (len(VLA_ORDER) + 2) + "}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
    ]
    for _, row in frame.iterrows():
        cells = [str(row["latex_name"])]
        for column in numeric_columns:
            value = row[column]
            if pd.isna(value):
                cells.append("")
                continue
            cell = "{:.4f}".format(float(value))
            if abs(float(value) - float(best_by_column[column])) < 1e-12:
                cell = r"\textbf{" + cell + "}"
            elif abs(float(value) - float(second_best_by_column[column])) < 1e-12:
                cell = r"\underline{" + cell + "}"
            cells.append(cell)
        cells.append("" if pd.isna(row["Rank"]) else str(int(row["Rank"])))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table*}",
        "",
    ])
    tex_path = output_dir / "rq2_baseline_performance.tex"
    atomic_text(tex_path, "\n".join(lines))
    return csv_path, tex_path


def statistics_table(comparisons: pd.DataFrame, output_dir: Path) -> Tuple[Path, Path]:
    frame = comparisons.sort_values(
        ["candidate_macro_roc_auc", "method_id"], ascending=[False, True]
    ).copy()
    csv_path = output_dir / "rq2_baseline_statistics.csv"
    write_csv(csv_path, frame)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{RQ2 -- Paired comparison of complete trajectory-level baselines against Harmonic Mean. The gap is baseline minus VoxelSBFL ROC-AUC; $p$-values use Holm correction and $r_{rb}$ is paired rank-biserial effect size.}",
        r"\label{tab:rq2-baseline-statistics}",
        r"{\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Baseline & ROC-AUC & Gap [95\% CI] & Holm $p$ & $r_{rb}$ & Equivalent \\",
        r"\midrule",
    ]
    for _, row in frame.iterrows():
        cells = [
            latex_escape(row["method"]),
            "{:.4f}".format(float(row["candidate_macro_roc_auc"])),
            "{:+.4f} [{:+.4f}, {:+.4f}]".format(
                float(row["auc_gap_candidate_minus_reference"]),
                float(row["gap_ci_low"]),
                float(row["gap_ci_high"]),
            ),
            "{:.4f}".format(float(row["holm_p"])),
            "{:+.3f}".format(float(row["rank_biserial"])),
            "Yes" if boolean(row["practically_equivalent"]) else "No",
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}", ""])
    tex_path = output_dir / "rq2_baseline_statistics.tex"
    atomic_text(tex_path, "\n".join(lines))
    return csv_path, tex_path


def availability_table(availability: pd.DataFrame, output_dir: Path) -> Tuple[Path, Path]:
    available_vlas = [vla for vla in VLA_ORDER if vla in set(availability["vla"].astype(str))]
    rows = []
    method_order = availability[["method_id", "method"]].drop_duplicates().sort_values("method_id")
    for _, method in method_order.iterrows():
        row = {"Method": method["method"]}
        for vla in VLA_ORDER:
            cell = availability[
                (availability["method_id"] == method["method_id"])
                & (availability["vla"] == vla)
            ]
            row[VLA_DISPLAY[vla]] = (
                float(cell.iloc[0]["availability_fraction"]) if len(cell) == 1 else float("nan")
            )
        rows.append(row)
    frame = pd.DataFrame(rows)
    csv_path = output_dir / "rq2_metric_availability.csv"
    write_csv(csv_path, frame)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{RQ2 -- Fraction of episodes with each stored trajectory-level metric. A method enters the primary comparison only when availability is 100\% in every selected cell. Blank cells denote VLA datasets that have not yet been executed.}",
        r"\label{tab:rq2-metric-availability}",
        r"{\scriptsize",
        r"\begin{tabular}{l" + "c" * len(VLA_ORDER) + "}",
        r"\toprule",
        " & ".join(["Metric", *[VLA_DISPLAY[vla] for vla in VLA_ORDER]]) + r" \\",
        r"\midrule",
    ]
    for _, row in frame.iterrows():
        cells = [latex_escape(row["Method"])]
        for vla in VLA_ORDER:
            value = row[VLA_DISPLAY[vla]]
            cells.append("" if pd.isna(value) else "{:.1f}\\%".format(100.0 * float(value)))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}", ""])
    tex_path = output_dir / "rq2_metric_availability.tex"
    atomic_text(tex_path, "\n".join(lines))
    return csv_path, tex_path


def main() -> int:
    args = parse_args()
    aggregate = args.aggregate_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    by_vla = read_csv(
        aggregate / "performance_by_vla.csv",
        {"vla", "method_id", "method", "family", "task_aware", "macro_roc_auc"},
    )
    overall = read_csv(
        aggregate / "overall_performance.csv",
        {"method_id", "method", "family", "task_aware", "macro_roc_auc", "overall_rank", "complete_primary_comparison"},
    )
    comparisons = read_csv(
        aggregate / "statistical_comparison.csv",
        {"method_id", "method", "candidate_macro_roc_auc", "auc_gap_candidate_minus_reference", "gap_ci_low", "gap_ci_high", "holm_p", "rank_biserial", "practically_equivalent"},
    )
    availability = read_csv(
        aggregate / "metric_availability.csv",
        {"vla", "method_id", "method", "availability_fraction"},
    )
    generated = [
        *performance_table(by_vla, overall, output),
        *statistics_table(comparisons, output),
        *availability_table(availability, output),
    ]
    try:
        from experiments.RQ2_baselines.generate_rq2_top5_rq1 import generate as generate_top5
    except ModuleNotFoundError:
        from generate_rq2_top5_rq1 import generate as generate_top5
    generated.extend(generate_top5(args.rq1_aggregate_dir, aggregate, output))
    manifest = {
        "rq1_aggregate_directory": str(args.rq1_aggregate_dir.expanduser().resolve()),
        "generated_files": [
            {"path": str(path.relative_to(repository_root())), "sha256": sha256_file(path)}
            for path in generated
        ],
        "aggregate_directory": str(aggregate),
    }
    atomic_text(output / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
