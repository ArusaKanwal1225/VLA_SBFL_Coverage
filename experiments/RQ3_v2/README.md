# RQ3 v2 execution guide

RQ3 v2 measures learning curves as the number of labelled trajectories used to
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
# Recompute aggregate files without refitting
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

Important aggregate outputs are:

- `cell_metrics.csv.gz`: every suite/formula result;
- `learning_curves_by_vla.csv`: ROC-AUC and coverage by VLA and suite size;
- `global_learning_curves.csv`: macro learning curves and rankings;
- `sample_efficiency_by_vla.csv`: NAULC, endpoint, gain, and 95% size;
- `global_sample_efficiency.csv`: overall sample-efficiency ranking;
- `adjacent_size_tests.csv`: matched Harmonic Mean learning-curve tests;
- `within_size_coverage_effects.csv`: block slopes after controlling suite size;
- `statistical_tests.json`: frozen confirmatory-test metadata;
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

The generated LaTeX tables require the `booktabs` package. Technique selection
for the compact main sample-efficiency table is reproducible: it includes the
union of techniques reaching a top-five NAULC rank for at least one VLA, plus
the frozen Harmonic Mean reference. Full 34-technique results remain available
in the supplementary tables.
