# Training-set size and coverage: paper RQ4

This directory is `experiments/RQ3_v2` in the source tree and implements RQ4 of the paper.

## Statistical analysis from aggregate results

Run these commands from the repository root with the dependencies declared in
`src_new/pyproject.toml` to calculate tests from the stored aggregate CSVs:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src_new python -B experiments/RQ3_v2/run_experiment.py --output-root experiments/RQ3_v2/results/learning_curves --statistical-review-only --wilcoxon-method exact
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src_new python -B experiments/RQ3_v2/generate_tables.py --aggregate-dir experiments/RQ3_v2/results/learning_curves/aggregate --output-dir experiments/RQ3_v2/results/learning_curves/tables --statistical-review-method exact
```

Use `approximate` in both commands to select the normal approximation.
Both modes write method-labelled review files and separate manifests; they do not
rebuild folds, load trajectories, refit models, or regenerate bootstrap intervals.
Historical files without method suffixes remain the original analysis.
The exact method reuses RQ1's conditional two-sided sign enumeration (zeros removed,
average ranks for absolute ties, maximum 20 nonzero pairs). Its inference requires
valid independent sign changes under the symmetric paired-difference null.
The original experimental runners retain their approximate default.

Holm correction is applied separately to four adjacent-size tests, 33
sample-efficiency comparisons and 34 conditional coverage-slope tests. The
reference row in the sample-efficiency CSV is descriptive and is not included
in its 33-test family.

The paired observations are the 16 VLA-by-generation-strategy blocks. Statistical
inference requires valid sign changes under the null; exact enumeration does not
establish block independence. This analysis does not compute bootstrap confidence
intervals. Meaningfulness labels combine observed mean differences with corrected
test decisions; they are not formal equivalence tests.

The training-set-size experiment measures learning curves as the number of labelled trajectories used to
construct the spectra increases. Within each repetition, the class-balanced
10/20/50/100/200-trajectory suites are nested. All evaluations use an untouched
outer run.

## Smoke test

From the repository root:

```bash
PYTHONPATH="$PWD:$PWD/src_new" python experiments/RQ3_v2/run_experiment.py --smoke
```

The smoke design runs all five suite sizes for one repetition, one fold, and
`EO1/DH/200`. It therefore checks nesting and learning-curve aggregation, not
statistical significance.

## Full primary experiment

This command executes experiment cells and requires the original input data.
Use a separate output directory for a new experiment. The aggregate-only commands
above reproduce the exact tests without executing experiment cells.

```bash
PYTHONPATH="$PWD:$PWD/src_new" python experiments/RQ3_v2/run_experiment.py
```

The default design fits 16,000 subset cells: 32 configurations, 10 held-out
folds, 10 nested-suite repetitions, and 5 suite sizes. Each fit evaluates all
34 techniques. Execution is deterministic and resumable; fitted models and
prediction tables are not saved unless explicitly requested.

Results are written under `experiments/RQ3_v2/results/learning_curves`.

## Useful options

```bash
# Rebuild original unsuffixed aggregate files from completed cells
PYTHONPATH="$PWD:$PWD/src_new" python experiments/RQ3_v2/run_experiment.py --analysis-only

# Generate the main and supplementary CSV/LaTeX tables
PYTHONPATH="$PWD:$PWD/src_new" python experiments/RQ3_v2/generate_tables.py

# Generate the main and supplementary PNG/PDF figures after aggregation
PYTHONPATH="$PWD:$PWD/src_new" python experiments/RQ3_v2/generate_plots.py

# Run a selected configuration with three repetitions
PYTHONPATH="$PWD:$PWD/src_new" python experiments/RQ3_v2/run_experiment.py \
  --configuration EO1__DH__200 --repetitions 3

# Retain the large optional artifacts
PYTHONPATH="$PWD:$PWD/src_new" python experiments/RQ3_v2/run_experiment.py \
  --save-models --save-predictions
```

The list below describes the default experiment outputs. The aggregate-only
commands above produce separate files with `_exact.csv` or `_approximate.csv`
suffixes and method-specific JSON metadata.

- `cell_metrics.csv.gz`: every suite/formula result;
- `learning_curves_by_vla.csv`: ROC-AUC and coverage by VLA and suite size;
- `global_learning_curves.csv`: macro learning curves and rankings;
- `sample_efficiency_by_vla.csv`: NAULC, endpoint, gain, and 95% size;
- `global_sample_efficiency.csv`: overall sample-efficiency ranking;
- `adjacent_size_tests.csv`: matched Harmonic Mean learning-curve tests;
- `within_size_coverage_effects.csv`: block slopes after controlling suite size;
- `statistical_tests.json`: metadata for the original statistical analysis;
- `RESULTS.md` and `summary.json`: human- and machine-readable summaries.

Publication artifacts are written to:

- `results/learning_curves/tables`: the three main-paper tables, complete
  VLA-specific supplementary tables for all 34 techniques, the conditional
  coverage table, and a provenance manifest;
- `results/learning_curves/figures/rq3v2_suite_size_effects.{pdf,png}`: the
  combined main-paper coverage and ROC-AUC learning-curve figure;
- `results/learning_curves/figures/rq3v2_supplementary_rank_heatmap.{pdf,png}`:
  the complete technique-rank heatmap;
- the original separate coverage, ROC-AUC, and unadjusted coverage--ROC-AUC
  figures, retained for flexible paper and supplementary layouts.

LaTeX tables require `booktabs`; the exact statistical tables also require
`longtable`. Figure generation additionally requires Matplotlib. Technique selection
for the compact main sample-efficiency table is reproducible: it includes the
union of techniques reaching a top-five NAULC rank for at least one VLA, plus
the Harmonic Mean reference. Full 34-technique results remain available
in the supplementary tables.
