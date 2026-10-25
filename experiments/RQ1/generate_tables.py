#!/usr/bin/env python3
"""Generate publication tables from the frozen RQ1 aggregate results.

The script deliberately depends only on the Python standard library. It reads
the aggregate artifacts produced by ``run_experiment.py`` and writes:

* the 34-technique cross-VLA ROC-AUC table;
* the inferential comparison for techniques reaching a VLA top-five; and
* one detailed supplementary table for each VLA model.

Both CSV and LaTeX versions are generated. All ranks are recomputed with ties
receiving the same minimum rank, so exactly equal techniques are not ordered
arbitrarily in the paper tables.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


EXPECTED_TECHNIQUE_COUNT = 34
VLA_ORDER = ("EO1", "Groot", "Pi0", "SpatialVLA")
VLA_DISPLAY_NAMES = {
    "EO1": "EO1",
    "Groot": "GR00T",
    "Pi0": "Pi0",
    "SpatialVLA": "SpatialVLA",
}
CONDITION_ORDER = ("DH", "DH_HybridDT", "DH_HybridRF", "DH_WeightedRF")
INPUT_FILES = (
    "overall_ranking.csv",
    "subgroup_metrics.csv",
    "statistical_comparison.csv",
    "primary_block_metrics.csv",
)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    root = repository_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aggregate-dir",
        type=Path,
        default=root / "experiments" / "RQ1" / "results" / "phase_a_primary" / "aggregate",
        help="Directory containing the frozen RQ1 aggregate CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "experiments" / "RQ1" / "results" / "phase_a_primary" / "tables",
        help="Directory in which generated CSV, LaTeX, and manifest files are written.",
    )
    parser.add_argument(
        "--top-rank-cutoff",
        type=int,
        default=5,
        help="Include a technique in the statistical table if it reaches this rank for any VLA.",
    )
    return parser.parse_args()


def read_csv(path: Path, required_columns: Iterable[str]) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or ())
        missing = set(required_columns) - columns
        if missing:
            raise ValueError("{} is missing columns: {}".format(path, ", ".join(sorted(missing))))
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError("{} contains no data rows".format(path))
    return rows


def finite_float(row: Mapping[str, str], field: str, context: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("{} has an invalid {} value".format(context, field)) from error
    if not math.isfinite(value):
        raise ValueError("{} has a non-finite {} value".format(context, field))
    return value


def integer(row: Mapping[str, str], field: str, context: str) -> int:
    value = finite_float(row, field, context)
    if not value.is_integer():
        raise ValueError("{} has a non-integer {} value".format(context, field))
    return int(value)


def boolean(row: Mapping[str, str], field: str, context: str) -> bool:
    value = row.get(field, "").strip().lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    raise ValueError("{} has an invalid {} value".format(context, field))


def keyed_rows(
    rows: Iterable[Mapping[str, str]],
    key_fields: Sequence[str],
    context: str,
) -> Dict[Tuple[str, ...], Mapping[str, str]]:
    result: Dict[Tuple[str, ...], Mapping[str, str]] = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        if key in result:
            raise ValueError("{} contains duplicate key {}".format(context, key))
        result[key] = row
    return result


def tie_aware_ranks(values: Mapping[str, float], tolerance: float = 1e-12) -> Dict[str, int]:
    """Return descending competition ranks (1, 1, 3, ...) with numeric ties."""

    ordered = sorted(values.items(), key=lambda item: (-item[1], item[0]))
    ranks: Dict[str, int] = {}
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and math.isclose(
            ordered[end][1], ordered[start][1], rel_tol=tolerance, abs_tol=tolerance
        ):
            end += 1
        rank = start + 1
        for formula_id, _ in ordered[start:end]:
            ranks[formula_id] = rank
        start = end
    return ranks


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


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, buffer.getvalue())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_auc_rank(value: float, rank: int, top_rank_cutoff: int) -> str:
    text = "{:.3f} ({})".format(value, rank)
    if rank == 1:
        return r"\textbf{" + text + "}"
    if rank <= top_rank_cutoff:
        return r"\underline{" + text + "}"
    return text


def validate_techniques(
    rows: Sequence[Mapping[str, str]],
    context: str,
) -> Dict[str, Mapping[str, str]]:
    by_id = {key[0]: row for key, row in keyed_rows(rows, ("formula_id",), context).items()}
    if len(by_id) != EXPECTED_TECHNIQUE_COUNT:
        raise ValueError(
            "{} must contain {} techniques, found {}".format(
                context, EXPECTED_TECHNIQUE_COUNT, len(by_id)
            )
        )
    for formula_id, row in by_id.items():
        if not formula_id.strip() or not row["technique"].strip():
            raise ValueError("{} contains an empty technique identifier or name".format(context))
    return by_id


def build_cross_vla_table(
    overall_by_id: Mapping[str, Mapping[str, str]],
    vla_by_id: Mapping[str, Mapping[str, Mapping[str, str]]],
    ordered_ids: Sequence[str],
    overall_ranks: Mapping[str, int],
    vla_ranks: Mapping[str, Mapping[str, int]],
    output_dir: Path,
    top_rank_cutoff: int,
) -> Tuple[Path, Path]:
    csv_rows: List[Dict[str, object]] = []
    for formula_id in ordered_ids:
        overall = overall_by_id[formula_id]
        row: Dict[str, object] = {
            "formula_id": formula_id,
            "technique": overall["technique"],
        }
        for vla in VLA_ORDER:
            subgroup = vla_by_id[vla][formula_id]
            display_name = VLA_DISPLAY_NAMES[vla]
            row["{}_roc_auc".format(display_name)] = "{:.6f}".format(
                finite_float(subgroup, "macro_roc_auc", "{} {}".format(vla, formula_id))
            )
            row["{}_rank".format(display_name)] = vla_ranks[vla][formula_id]
        row["overall_roc_auc"] = "{:.6f}".format(
            finite_float(overall, "macro_roc_auc", "overall {}".format(formula_id))
        )
        row["overall_rank"] = overall_ranks[formula_id]
        csv_rows.append(row)

    csv_fields = ["formula_id", "technique"]
    for vla in VLA_ORDER:
        display_name = VLA_DISPLAY_NAMES[vla]
        csv_fields.extend(
            ["{}_roc_auc".format(display_name), "{}_rank".format(display_name)]
        )
    csv_fields.extend(["overall_roc_auc", "overall_rank"])
    csv_path = output_dir / "rq1_cross_vla_roc_auc.csv"
    write_csv(csv_path, csv_rows, csv_fields)

    latex_rows = []
    for formula_id in ordered_ids:
        cells = [latex_escape(overall_by_id[formula_id]["technique"])]
        for vla in VLA_ORDER:
            cells.append(
                format_auc_rank(
                    finite_float(
                        vla_by_id[vla][formula_id],
                        "macro_roc_auc",
                        "{} {}".format(vla, formula_id),
                    ),
                    vla_ranks[vla][formula_id],
                    top_rank_cutoff,
                )
            )
        cells.append(
            format_auc_rank(
                finite_float(
                    overall_by_id[formula_id],
                    "macro_roc_auc",
                    "overall {}".format(formula_id),
                ),
                overall_ranks[formula_id],
                top_rank_cutoff,
            )
        )
        latex_rows.append(" & ".join(cells) + r" \\")

    header = (
        "Technique & "
        + " & ".join(latex_escape(VLA_DISPLAY_NAMES[vla]) for vla in VLA_ORDER)
        + r" & Overall \\"
    )
    latex = "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{Macro ROC-AUC of the 34 SBFL techniques for each VLA model. "
            r"Parentheses report tie-aware ranks. Bold denotes rank 1 and underlining "
            r"denotes the remaining top-five ranks. Harmonic Mean and Rogot2 are tied "
            r"because they produce identical results in every reported column.}",
            r"\label{tab:rq1-cross-vla}",
            r"{\scriptsize",
            r"\setlength{\tabcolsep}{4pt}",
            r"\renewcommand{\arraystretch}{0.96}",
            r"\begin{tabular}{lccccc}",
            r"\toprule",
            header,
            r"\midrule",
            *latex_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table*}",
            "",
        ]
    )
    latex_path = output_dir / "rq1_cross_vla_roc_auc.tex"
    atomic_text(latex_path, latex)
    return csv_path, latex_path


def p_value_text(value: float) -> str:
    if value < 0.001:
        return r"$<0.001$"
    return "{:.3f}".format(value)


def build_statistical_table(
    statistics_by_id: Mapping[str, Mapping[str, str]],
    overall_by_id: Mapping[str, Mapping[str, str]],
    ordered_ids: Sequence[str],
    overall_ranks: Mapping[str, int],
    vla_ranks: Mapping[str, Mapping[str, int]],
    output_dir: Path,
    top_rank_cutoff: int,
) -> Tuple[Path, Path, List[str]]:
    competitive_ids = [
        formula_id
        for formula_id in ordered_ids
        if any(vla_ranks[vla][formula_id] <= top_rank_cutoff for vla in VLA_ORDER)
    ]
    if not competitive_ids:
        raise ValueError("the competitive technique set is empty")
    reference_id = ordered_ids[0]

    csv_rows: List[Dict[str, object]] = []
    latex_rows: List[str] = []
    for formula_id in competitive_ids:
        stats = statistics_by_id[formula_id]
        context = "statistics {}".format(formula_id)
        auc = finite_float(stats, "macro_roc_auc", context)
        auc_lower = finite_float(stats, "roc_auc_ci_lower", context)
        auc_upper = finite_float(stats, "roc_auc_ci_upper", context)
        gap = finite_float(stats, "delta_auc_vs_leader", context)
        gap_lower = finite_float(stats, "delta_vs_leader_ci_lower", context)
        gap_upper = finite_float(stats, "delta_vs_leader_ci_upper", context)
        adjusted_p = finite_float(stats, "holm_adjusted_p_value", context)
        effect = finite_float(stats, "rank_biserial_leader_minus_candidate", context)
        equivalent = boolean(stats, "practically_equivalent", context)
        top_vlas = [vla for vla in VLA_ORDER if vla_ranks[vla][formula_id] <= top_rank_cutoff]
        csv_rows.append(
            {
                "formula_id": formula_id,
                "technique": overall_by_id[formula_id]["technique"],
                "overall_rank": overall_ranks[formula_id],
                "overall_roc_auc": "{:.6f}".format(auc),
                "roc_auc_ci_lower": "{:.6f}".format(auc_lower),
                "roc_auc_ci_upper": "{:.6f}".format(auc_upper),
                "gap_to_leader": "{:.6f}".format(gap),
                "gap_ci_lower": "{:.6f}".format(gap_lower),
                "gap_ci_upper": "{:.6f}".format(gap_upper),
                "holm_adjusted_p_value": "{:.6g}".format(adjusted_p),
                "rank_biserial_leader_minus_candidate": "{:.6f}".format(effect),
                "practically_equivalent": str(equivalent).lower(),
                "top_five_vlas": ";".join(top_vlas),
                "is_reference": str(formula_id == reference_id).lower(),
            }
        )

        technique = latex_escape(overall_by_id[formula_id]["technique"])
        if overall_ranks[formula_id] == 1:
            technique = r"\textbf{" + technique + "}"
        auc_cell = "{:.3f} [{:.3f}, {:.3f}]".format(auc, auc_lower, auc_upper)
        if formula_id == reference_id:
            gap_cell = "Ref."
            p_cell = r"--"
            effect_cell = r"--"
            equivalent_cell = "Ref."
        else:
            gap_cell = "{:.3f} [{:.3f}, {:.3f}]".format(gap, gap_lower, gap_upper)
            p_cell = p_value_text(adjusted_p)
            effect_cell = "{:.3f}".format(effect)
            equivalent_cell = "Yes" if equivalent else "No"
        latex_rows.append(
            " & ".join(
                [technique, auc_cell, gap_cell, p_cell, effect_cell, equivalent_cell]
            )
            + r" \\"
        )

    csv_fields = (
        "formula_id",
        "technique",
        "overall_rank",
        "overall_roc_auc",
        "roc_auc_ci_lower",
        "roc_auc_ci_upper",
        "gap_to_leader",
        "gap_ci_lower",
        "gap_ci_upper",
        "holm_adjusted_p_value",
        "rank_biserial_leader_minus_candidate",
        "practically_equivalent",
        "top_five_vlas",
        "is_reference",
    )
    csv_path = output_dir / "rq1_competitive_statistics.csv"
    write_csv(csv_path, csv_rows, csv_fields)

    latex = "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{Statistical comparison of the techniques reaching a top-five "
            r"rank for at least one VLA. The AUC gap is the leader's ROC-AUC minus the "
            r"candidate's ROC-AUC. Confidence intervals are 95\%; $p$-values use Holm "
            r"correction and $r_{rb}$ is the paired rank-biserial effect size.}",
            r"\label{tab:rq1-competitive-statistics}",
            r"{\scriptsize",
            r"\setlength{\tabcolsep}{4pt}",
            r"\begin{tabular}{lccccc}",
            r"\toprule",
            r"Technique & ROC-AUC [95\% CI] & Gap [95\% CI] & Holm $p$ & $r_{rb}$ & Equivalent \\",
            r"\midrule",
            *latex_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table*}",
            "",
        ]
    )
    latex_path = output_dir / "rq1_competitive_statistics.tex"
    atomic_text(latex_path, latex)
    return csv_path, latex_path, competitive_ids


def sample_standard_deviation(values: Sequence[float]) -> float:
    if len(values) < 2:
        raise ValueError("at least two values are required for a sample standard deviation")
    return float(statistics.stdev(values))


def build_supplementary_tables(
    vla_by_id: Mapping[str, Mapping[str, Mapping[str, str]]],
    primary_blocks: Sequence[Mapping[str, str]],
    ordered_ids: Sequence[str],
    vla_ranks: Mapping[str, Mapping[str, int]],
    output_dir: Path,
) -> List[Path]:
    block_map: MutableMapping[Tuple[str, str], List[Mapping[str, str]]] = {}
    for row in primary_blocks:
        key = (row["vla"], row["formula_id"])
        block_map.setdefault(key, []).append(row)

    outputs: List[Path] = []
    for vla in VLA_ORDER:
        subgroup = vla_by_id[vla]
        ap_values = {
            formula_id: finite_float(row, "macro_average_precision", "{} {}".format(vla, formula_id))
            for formula_id, row in subgroup.items()
        }
        ap_ranks = tie_aware_ranks(ap_values)
        csv_rows: List[Dict[str, object]] = []
        latex_rows: List[str] = []
        for formula_id in ordered_ids:
            blocks = block_map.get((vla, formula_id), [])
            conditions = tuple(sorted((row["condition"] for row in blocks), key=CONDITION_ORDER.index))
            if conditions != CONDITION_ORDER:
                raise ValueError(
                    "{} {} expected conditions {}, found {}".format(
                        vla, formula_id, CONDITION_ORDER, conditions
                    )
                )
            auc_condition_values = [
                finite_float(row, "roc_auc", "{} {} primary block".format(vla, formula_id))
                for row in blocks
            ]
            ap_condition_values = [
                finite_float(
                    row,
                    "average_precision",
                    "{} {} primary block".format(vla, formula_id),
                )
                for row in blocks
            ]
            macro_auc = finite_float(
                subgroup[formula_id], "macro_roc_auc", "{} {}".format(vla, formula_id)
            )
            macro_ap = ap_values[formula_id]
            if not math.isclose(macro_auc, statistics.mean(auc_condition_values), abs_tol=1e-12):
                raise ValueError("{} {} ROC-AUC does not reconcile to primary blocks".format(vla, formula_id))
            if not math.isclose(macro_ap, statistics.mean(ap_condition_values), abs_tol=1e-12):
                raise ValueError("{} {} AP does not reconcile to primary blocks".format(vla, formula_id))
            auc_sd = sample_standard_deviation(auc_condition_values)
            ap_sd = sample_standard_deviation(ap_condition_values)
            csv_rows.append(
                {
                    "formula_id": formula_id,
                    "technique": subgroup[formula_id]["technique"],
                    "macro_roc_auc": "{:.6f}".format(macro_auc),
                    "roc_auc_rank": vla_ranks[vla][formula_id],
                    "roc_auc_condition_sd": "{:.6f}".format(auc_sd),
                    "macro_average_precision": "{:.6f}".format(macro_ap),
                    "average_precision_rank": ap_ranks[formula_id],
                    "average_precision_condition_sd": "{:.6f}".format(ap_sd),
                    "conditions": len(blocks),
                }
            )
            latex_rows.append(
                " & ".join(
                    [
                        latex_escape(subgroup[formula_id]["technique"]),
                        "{:.3f}".format(macro_auc),
                        str(vla_ranks[vla][formula_id]),
                        "{:.3f}".format(auc_sd),
                        "{:.3f}".format(macro_ap),
                        str(ap_ranks[formula_id]),
                        "{:.3f}".format(ap_sd),
                    ]
                )
                + r" \\"
            )

        fields = (
            "formula_id",
            "technique",
            "macro_roc_auc",
            "roc_auc_rank",
            "roc_auc_condition_sd",
            "macro_average_precision",
            "average_precision_rank",
            "average_precision_condition_sd",
            "conditions",
        )
        display_name = VLA_DISPLAY_NAMES[vla]
        csv_path = output_dir / "rq1_supplementary_{}.csv".format(display_name)
        write_csv(csv_path, csv_rows, fields)
        outputs.append(csv_path)

        latex = "\n".join(
            [
                r"\begin{table*}[t]",
                r"\centering",
                r"\caption{Detailed RQ1 results for " + latex_escape(display_name) +
                r". SD reports variation across the four trajectory-generation conditions.}",
                r"\label{tab:rq1-supplementary-" + display_name.lower() + "}",
                r"{\scriptsize",
                r"\setlength{\tabcolsep}{4pt}",
                r"\renewcommand{\arraystretch}{0.96}",
                r"\begin{tabular}{lrrrrrr}",
                r"\toprule",
                r"Technique & ROC-AUC & Rank & AUC SD & AP & AP rank & AP SD \\",
                r"\midrule",
                *latex_rows,
                r"\bottomrule",
                r"\end{tabular}",
                r"}",
                r"\end{table*}",
                "",
            ]
        )
        latex_path = output_dir / "rq1_supplementary_{}.tex".format(display_name)
        atomic_text(latex_path, latex)
        outputs.append(latex_path)
    return outputs


def generate_tables(aggregate_dir: Path, output_dir: Path, top_rank_cutoff: int) -> List[Path]:
    if top_rank_cutoff < 1:
        raise ValueError("top-rank-cutoff must be positive")
    aggregate_dir = aggregate_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()

    overall_rows = read_csv(
        aggregate_dir / "overall_ranking.csv",
        ("formula_id", "technique", "macro_roc_auc", "overall_rank"),
    )
    overall_by_id = validate_techniques(overall_rows, "overall_ranking.csv")
    formula_ids = set(overall_by_id)

    subgroup_rows = read_csv(
        aggregate_dir / "subgroup_metrics.csv",
        (
            "dimension",
            "group",
            "formula_id",
            "technique",
            "macro_roc_auc",
            "macro_average_precision",
            "group_rank",
        ),
    )
    vla_rows = [row for row in subgroup_rows if row["dimension"] == "vla"]
    vla_index = keyed_rows(vla_rows, ("group", "formula_id"), "VLA subgroup rows")
    vla_by_id: Dict[str, Dict[str, Mapping[str, str]]] = {}
    for vla in VLA_ORDER:
        vla_by_id[vla] = {
            formula_id: row
            for (group, formula_id), row in vla_index.items()
            if group == vla
        }
        if set(vla_by_id[vla]) != formula_ids:
            raise ValueError("{} subgroup does not contain the same 34 techniques".format(vla))
    unexpected_vlas = {group for group, _ in vla_index} - set(VLA_ORDER)
    if unexpected_vlas:
        raise ValueError("unexpected VLA subgroups: {}".format(", ".join(sorted(unexpected_vlas))))

    for vla, rows in vla_by_id.items():
        for formula_id, row in rows.items():
            if row["technique"] != overall_by_id[formula_id]["technique"]:
                raise ValueError("{} {} has an inconsistent technique name".format(vla, formula_id))

    overall_values = {
        formula_id: finite_float(row, "macro_roc_auc", "overall {}".format(formula_id))
        for formula_id, row in overall_by_id.items()
    }
    overall_ranks = tie_aware_ranks(overall_values)
    ordered_ids = sorted(
        formula_ids,
        key=lambda formula_id: (
            overall_ranks[formula_id],
            integer(
                overall_by_id[formula_id],
                "overall_rank",
                "overall {}".format(formula_id),
            ),
            overall_by_id[formula_id]["technique"],
        ),
    )
    vla_ranks: Dict[str, Dict[str, int]] = {}
    for vla in VLA_ORDER:
        values = {
            formula_id: finite_float(row, "macro_roc_auc", "{} {}".format(vla, formula_id))
            for formula_id, row in vla_by_id[vla].items()
        }
        vla_ranks[vla] = tie_aware_ranks(values)
        for formula_id, row in vla_by_id[vla].items():
            source_rank = integer(row, "group_rank", "{} {}".format(vla, formula_id))
            if source_rank != vla_ranks[vla][formula_id]:
                raise ValueError(
                    "{} {} source rank {} does not match recomputed rank {}".format(
                        vla, formula_id, source_rank, vla_ranks[vla][formula_id]
                    )
                )

    statistics_rows = read_csv(
        aggregate_dir / "statistical_comparison.csv",
        (
            "formula_id",
            "technique",
            "macro_roc_auc",
            "delta_auc_vs_leader",
            "holm_adjusted_p_value",
            "rank_biserial_leader_minus_candidate",
            "practically_equivalent",
            "roc_auc_ci_lower",
            "roc_auc_ci_upper",
            "delta_vs_leader_ci_lower",
            "delta_vs_leader_ci_upper",
        ),
    )
    statistics_by_id = validate_techniques(statistics_rows, "statistical_comparison.csv")
    if set(statistics_by_id) != formula_ids:
        raise ValueError("statistical comparison does not contain the same techniques")

    primary_blocks = read_csv(
        aggregate_dir / "primary_block_metrics.csv",
        ("vla", "condition", "formula_id", "technique", "roc_auc", "average_precision"),
    )
    expected_primary_rows = len(VLA_ORDER) * len(CONDITION_ORDER) * EXPECTED_TECHNIQUE_COUNT
    if len(primary_blocks) != expected_primary_rows:
        raise ValueError(
            "primary_block_metrics.csv expected {} rows, found {}".format(
                expected_primary_rows, len(primary_blocks)
            )
        )
    keyed_rows(
        primary_blocks,
        ("vla", "condition", "formula_id"),
        "primary_block_metrics.csv",
    )

    outputs: List[Path] = []
    outputs.extend(
        build_cross_vla_table(
            overall_by_id,
            vla_by_id,
            ordered_ids,
            overall_ranks,
            vla_ranks,
            output_dir,
            top_rank_cutoff,
        )
    )
    statistics_csv, statistics_tex, competitive_ids = build_statistical_table(
        statistics_by_id,
        overall_by_id,
        ordered_ids,
        overall_ranks,
        vla_ranks,
        output_dir,
        top_rank_cutoff,
    )
    outputs.extend((statistics_csv, statistics_tex))
    outputs.extend(
        build_supplementary_tables(
            vla_by_id,
            primary_blocks,
            ordered_ids,
            vla_ranks,
            output_dir,
        )
    )

    try:
        source_directory = str(aggregate_dir.relative_to(repository_root()))
    except ValueError:
        source_directory = str(aggregate_dir)
    manifest = {
        "source_directory": source_directory,
        "source_sha256": {
            filename: sha256_file(aggregate_dir / filename) for filename in INPUT_FILES
        },
        "technique_count": EXPECTED_TECHNIQUE_COUNT,
        "vla_order": list(VLA_ORDER),
        "condition_order": list(CONDITION_ORDER),
        "top_rank_cutoff": top_rank_cutoff,
        "competitive_formula_ids": competitive_ids,
        "rank_policy": "descending competition rank with numeric ties",
        "supplementary_sd_policy": "sample SD across four trajectory-generation conditions",
        "outputs": [path.name for path in outputs],
    }
    manifest_path = output_dir / "manifest.json"
    atomic_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    outputs.append(manifest_path)
    return outputs


def main() -> None:
    args = parse_args()
    outputs = generate_tables(args.aggregate_dir, args.output_dir, args.top_rank_cutoff)
    print("Generated {} RQ1 table artifacts in {}".format(len(outputs), args.output_dir.resolve()))
    for path in outputs:
        print("- {}".format(path.resolve()))


if __name__ == "__main__":
    main()
