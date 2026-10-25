#!/usr/bin/env python3
"""Generate the main-paper and supplementary RQ3 v2 figures."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
VLA_ORDER = ("EO1", "Groot", "Pi0", "SpatialVLA")
VLA_DISPLAY = {"EO1": "EO1", "Groot": "GR00T", "Pi0": "Pi0", "SpatialVLA": "SpatialVLA"}
REFERENCE_FORMULA = "harmonic_mean"
TRAINING_SIZES = (10, 20, 50, 100, 200)
EXPECTED_RANK_MAX = 34


def apply_publication_style() -> None:
    """Use a restrained, vector-friendly style shared by every output."""

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def select_techniques(efficiency: pd.DataFrame, count: int) -> Sequence[str]:
    primary = efficiency[np.isclose(efficiency["voxel_size"].astype(float), 0.02)]
    ordered = list(primary.sort_values(
        ["naulc_rank", "formula_id"], kind="mergesort")["formula_id"].astype(str))
    selected = ordered[:count]
    if REFERENCE_FORMULA not in selected:
        selected.append(REFERENCE_FORMULA)
    return tuple(dict.fromkeys(selected))


def technique_labels(frame: pd.DataFrame) -> dict[str, str]:
    return dict(zip(frame["formula_id"].astype(str), frame["technique"].astype(str)))


def subplot_grid(title: str):
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True)
    figure.suptitle(title, fontsize=13)
    return figure, axes.ravel()


def finish_axes(axes: Iterable[plt.Axes], training_sizes: Sequence[int]) -> None:
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xticks(training_sizes, labels=[str(value) for value in training_sizes])
        axis.grid(axis="y", color="0.88", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)


def save_figure(
    figure: plt.Figure,
    output_directory: Path,
    stem: str,
    dpi: int,
    bottom: float = 0.11,
) -> None:
    figure.tight_layout(rect=(0.0, bottom, 1.0, 0.96))
    figure.savefig(output_directory / (stem + ".png"), dpi=dpi, bbox_inches="tight")
    figure.savefig(output_directory / (stem + ".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_combined_suite_size_effects(
    curves: pd.DataFrame,
    selected: Sequence[str],
    output_directory: Path,
    dpi: int,
) -> None:
    """Create the main two-part figure linking suite size, coverage, and AUC."""

    primary = curves[np.isclose(curves["voxel_size"].astype(float), 0.02)]
    reference = primary[
        primary["formula_id"].astype(str) == REFERENCE_FORMULA
    ]
    performance = primary[primary["formula_id"].astype(str).isin(selected)]
    labels = technique_labels(performance)
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.8, len(selected)))
    color_by_formula = dict(zip(selected, colors))
    markers = ("o", "s", "^", "D", "P", "X")
    marker_by_formula = {
        formula_id: markers[index % len(markers)]
        for index, formula_id in enumerate(selected)
    }

    figure, axes = plt.subplots(
        2, 4, figsize=(13.8, 6.3), sharex="col", sharey="row"
    )
    for column, vla in enumerate(VLA_ORDER):
        coverage_axis = axes[0, column]
        coverage = reference[reference["vla"].astype(str) == vla].sort_values(
            "training_size", kind="mergesort"
        )
        coverage_axis.plot(
            coverage["training_size"],
            100.0 * coverage["test_unique_coverage"],
            marker="s",
            linewidth=1.7,
            label="Held-out coverage",
        )
        coverage_axis.plot(
            coverage["training_size"],
            100.0 * coverage["calibration_unique_coverage"],
            marker="o",
            markerfacecolor="white",
            markeredgewidth=1.2,
            linestyle="--",
            linewidth=1.5,
            zorder=3,
            label="Calibration coverage",
        )
        coverage_axis.set_title(VLA_DISPLAY[vla])
        if column == 0:
            coverage_axis.set_ylabel("Unique-voxel coverage (%)")

        performance_axis = axes[1, column]
        vla_performance = performance[performance["vla"].astype(str) == vla]
        for formula_id in selected:
            series = vla_performance[
                vla_performance["formula_id"].astype(str) == formula_id
            ].sort_values("training_size", kind="mergesort")
            performance_axis.plot(
                series["training_size"],
                series["roc_auc"],
                marker=marker_by_formula[formula_id],
                markersize=3.6,
                linewidth=2.0 if formula_id == REFERENCE_FORMULA else 1.25,
                color=color_by_formula[formula_id],
                label=labels.get(formula_id, formula_id),
            )
        performance_axis.set_xlabel("Training trajectories")
        if column == 0:
            performance_axis.set_ylabel("Held-out ROC-AUC")

    for axis in axes.ravel():
        axis.set_xscale("log", base=2)
        axis.set_xticks(TRAINING_SIZES, labels=[str(value) for value in TRAINING_SIZES])
        axis.grid(axis="y", color="0.88", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)

    figure.text(0.01, 0.975, "(a) Voxel coverage", fontsize=10, weight="semibold")
    figure.text(0.01, 0.485, "(b) Failure-discrimination performance", fontsize=10,
                weight="semibold")
    coverage_handles, coverage_labels = axes[0, 0].get_legend_handles_labels()
    coverage_handles = coverage_handles[::-1]
    coverage_labels = coverage_labels[::-1]
    performance_handles, performance_labels = axes[1, 0].get_legend_handles_labels()
    figure.legend(
        coverage_handles,
        coverage_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.525),
        ncol=2,
        frameon=False,
    )
    figure.legend(
        performance_handles,
        performance_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=min(5, len(performance_handles)),
        frameon=False,
    )
    figure.subplots_adjust(
        left=0.065, right=0.995, top=0.91, bottom=0.13, wspace=0.12, hspace=0.52
    )
    figure.savefig(
        output_directory / "rq3v2_suite_size_effects.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    figure.savefig(
        output_directory / "rq3v2_suite_size_effects.pdf", bbox_inches="tight"
    )
    plt.close(figure)


def rank_matrix(
    curves: pd.DataFrame, vla: str, formula_order: Sequence[str]
) -> np.ndarray:
    """Return the complete technique-by-size rank matrix for one VLA."""

    frame = curves[
        np.isclose(curves["voxel_size"].astype(float), 0.02)
        & (curves["vla"].astype(str) == vla)
    ]
    pivot = frame.pivot(
        index="formula_id", columns="training_size", values="roc_auc_rank"
    ).reindex(index=list(formula_order), columns=list(TRAINING_SIZES))
    if pivot.isna().any().any():
        raise ValueError("incomplete rank matrix for {}".format(vla))
    return pivot.to_numpy(dtype=float)


def plot_rank_heatmap(
    curves: pd.DataFrame,
    efficiency: pd.DataFrame,
    output_directory: Path,
    dpi: int,
) -> None:
    """Plot all 34 technique ranks across suite sizes and VLA models."""

    primary_efficiency = efficiency[
        np.isclose(efficiency["voxel_size"].astype(float), 0.02)
    ].sort_values(["naulc_rank", "formula_id"], kind="mergesort")
    formula_order = tuple(primary_efficiency["formula_id"].astype(str))
    labels = technique_labels(primary_efficiency)
    technique_names = [labels[formula_id] for formula_id in formula_order]
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 14.0), sharex=True, sharey=True)
    image_artist = None
    for axis, vla in zip(axes.ravel(), VLA_ORDER):
        matrix = rank_matrix(curves, vla, formula_order)
        image_artist = axis.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            cmap="cividis_r",
            vmin=1,
            vmax=EXPECTED_RANK_MAX,
        )
        axis.set_title(VLA_DISPLAY[vla])
        axis.set_xticks(range(len(TRAINING_SIZES)), labels=[str(v) for v in TRAINING_SIZES])
        axis.set_yticks(range(len(technique_names)), labels=technique_names)
        axis.set_xlabel("Training trajectories")
        for row_index, column_index in np.argwhere(matrix <= 5.0):
            rank = matrix[row_index, column_index]
            text = str(int(round(rank))) if np.isclose(rank, round(rank)) else "{:.1f}".format(rank)
            axis.text(
                column_index,
                row_index,
                text,
                ha="center",
                va="center",
                fontsize=6.5,
                color="black",
                weight="semibold",
            )
    if image_artist is None:
        raise ValueError("no rank data available")
    colorbar_axis = figure.add_axes((0.925, 0.25, 0.015, 0.50))
    colorbar = figure.colorbar(image_artist, cax=colorbar_axis)
    colorbar.set_label("ROC-AUC rank (1 = best)")
    colorbar.set_ticks((1, 5, 10, 15, 20, 25, 30, 34))
    figure.suptitle(
        "Technique ranks across training-suite sizes", fontsize=13, y=0.995
    )
    figure.subplots_adjust(
        left=0.19, right=0.90, top=0.965, bottom=0.045, wspace=0.13, hspace=0.13
    )
    figure.savefig(
        output_directory / "rq3v2_supplementary_rank_heatmap.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    figure.savefig(
        output_directory / "rq3v2_supplementary_rank_heatmap.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_roc_auc_learning_curves(
    curves: pd.DataFrame,
    selected: Sequence[str],
    output_directory: Path,
    dpi: int,
) -> None:
    primary = curves[
        np.isclose(curves["voxel_size"].astype(float), 0.02)
        & curves["formula_id"].astype(str).isin(selected)
    ]
    labels = technique_labels(primary)
    sizes = sorted(primary["training_size"].astype(int).unique())
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.8, len(selected)))
    color_by_formula = dict(zip(selected, colors))
    figure, axes = subplot_grid("Held-out ROC-AUC learning curves")
    for axis, vla in zip(axes, VLA_ORDER):
        frame = primary[primary["vla"].astype(str) == vla]
        for formula_id in selected:
            series = frame[frame["formula_id"].astype(str) == formula_id].sort_values(
                "training_size", kind="mergesort")
            if series.empty:
                continue
            axis.plot(
                series["training_size"], series["roc_auc"], marker="o", markersize=3.5,
                linewidth=2.1 if formula_id == REFERENCE_FORMULA else 1.35,
                color=color_by_formula[formula_id], label=labels.get(formula_id, formula_id),
            )
        axis.set_title(VLA_DISPLAY[vla])
        axis.set_ylabel("ROC-AUC")
        axis.set_xlabel("Spectrum-construction trajectories")
    finish_axes(axes, sizes)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, legend_labels, loc="lower center", ncol=min(3, len(handles)),
                      frameon=False, bbox_to_anchor=(0.5, 0.005))
    save_figure(figure, output_directory, "roc_auc_vs_training_size", dpi)


def plot_coverage_learning_curves(
    curves: pd.DataFrame,
    output_directory: Path,
    dpi: int,
) -> None:
    primary = curves[
        np.isclose(curves["voxel_size"].astype(float), 0.02)
        & (curves["formula_id"].astype(str) == REFERENCE_FORMULA)
    ]
    sizes = sorted(primary["training_size"].astype(int).unique())
    figure, axes = subplot_grid("Voxel coverage as the trajectory suite grows")
    for axis, vla in zip(axes, VLA_ORDER):
        frame = primary[primary["vla"].astype(str) == vla].sort_values(
            "training_size", kind="mergesort")
        if not frame.empty:
            axis.plot(
                frame["training_size"], 100.0 * frame["calibration_unique_coverage"],
                marker="o", linewidth=1.8, label="Calibration coverage")
            axis.plot(
                frame["training_size"], 100.0 * frame["test_unique_coverage"],
                marker="s", linewidth=1.8, label="Held-out coverage")
        axis.set_title(VLA_DISPLAY[vla])
        axis.set_ylabel("Unique-voxel coverage (%)")
        axis.set_xlabel("Spectrum-construction trajectories")
    finish_axes(axes, sizes)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, legend_labels, loc="lower center", ncol=2,
                      frameon=False, bbox_to_anchor=(0.5, 0.005))
    save_figure(figure, output_directory, "coverage_vs_training_size", dpi)


def plot_auc_against_coverage(
    curves: pd.DataFrame,
    selected: Sequence[str],
    output_directory: Path,
    dpi: int,
) -> None:
    primary = curves[
        np.isclose(curves["voxel_size"].astype(float), 0.02)
        & curves["formula_id"].astype(str).isin(selected)
    ]
    labels = technique_labels(primary)
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.8, len(selected)))
    color_by_formula = dict(zip(selected, colors))
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    figure.suptitle("ROC-AUC versus achieved calibration coverage", fontsize=13)
    for axis, vla in zip(axes.ravel(), VLA_ORDER):
        frame = primary[primary["vla"].astype(str) == vla]
        for formula_id in selected:
            series = frame[frame["formula_id"].astype(str) == formula_id].sort_values(
                "calibration_unique_coverage", kind="mergesort")
            if series.empty:
                continue
            axis.plot(
                100.0 * series["calibration_unique_coverage"], series["roc_auc"],
                marker="o", markersize=3.5,
                linewidth=2.1 if formula_id == REFERENCE_FORMULA else 1.35,
                color=color_by_formula[formula_id], label=labels.get(formula_id, formula_id),
            )
            if formula_id == REFERENCE_FORMULA:
                for row in series.itertuples(index=False):
                    axis.annotate(
                        str(int(row.training_size)),
                        (100.0 * row.calibration_unique_coverage, row.roc_auc),
                        xytext=(4, 2), textcoords="offset points", fontsize=7, color="0.35")
        axis.set_title(VLA_DISPLAY[vla])
        axis.set_xlabel("Calibration unique-voxel coverage (%)")
        axis.set_ylabel("ROC-AUC")
        axis.grid(color="0.88", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    handles, legend_labels = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, legend_labels, loc="lower center", ncol=min(3, len(handles)),
                      frameon=False, bbox_to_anchor=(0.5, 0.005))
    save_figure(figure, output_directory, "roc_auc_vs_calibration_coverage", dpi)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-root", type=Path,
        default=Path("experiments/RQ3_v2/results/learning_curves"))
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--top-techniques", type=int, default=5)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.top_techniques < 1 or args.dpi < 72:
        raise ValueError("top-techniques must be positive and dpi must be at least 72")
    apply_publication_style()
    result_root = args.result_root.expanduser().resolve()
    aggregate = result_root / "aggregate"
    output_directory = (
        args.output_directory.expanduser().resolve()
        if args.output_directory else result_root / "figures")
    output_directory.mkdir(parents=True, exist_ok=True)
    curves = pd.read_csv(aggregate / "learning_curves_by_vla.csv")
    efficiency = pd.read_csv(aggregate / "global_sample_efficiency.csv")
    selected = select_techniques(efficiency, args.top_techniques)
    plot_combined_suite_size_effects(
        curves, selected, output_directory, args.dpi)
    plot_roc_auc_learning_curves(curves, selected, output_directory, args.dpi)
    plot_coverage_learning_curves(curves, output_directory, args.dpi)
    plot_auc_against_coverage(curves, selected, output_directory, args.dpi)
    plot_rank_heatmap(curves, efficiency, output_directory, args.dpi)
    print("Wrote RQ3 v2 figures to {}".format(output_directory))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
