# RQ1 table generation

For the exact pairwise statistical results, use the explicit commands in
"Exact pairwise comparisons" below. The default reporting command reads the
original unsuffixed aggregate files, including their original statistical results.

Run the reporting script after the RQ1 aggregate results have been generated:

```bash
python experiments/RQ1/generate_tables.py
```

By default, the script reads `results/phase_a_primary/aggregate` and writes the
publication tables to `results/phase_a_primary/tables`. Use `--aggregate-dir`
and `--output-dir` to select another completed RQ1 run.

The generated artifacts comprise:

- one CSV and one LaTeX table comparing all 34 techniques across the four VLA
  models and overall;
- one CSV and one LaTeX inferential table containing the union of techniques
  ranked in the top five for at least one VLA; and
- one detailed CSV and one detailed LaTeX table per VLA, including ROC-AUC,
  average precision, ranks, and variation across trajectory-generation
  conditions.

The script validates the source schemas and experiment dimensions, reconciles
the VLA summaries against the primary blocks, and recomputes tie-aware ranks.
The generated manifest records source hashes and reporting choices.

## Exact pairwise comparisons

Generate the statistical results from `primary_block_metrics.csv`, without
loading trajectories or executing folds. Run commands from the repository root
with the dependencies declared in `src_new/pyproject.toml`:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src_new python -B experiments/RQ1/run_experiment.py \
  --analysis-only --wilcoxon-method exact
```

This writes method-labelled all-pairs and Harmonic Mean reference comparisons,
plus the descriptive leave-one-factor-out and within-factor ranking sensitivity
artifacts in the `aggregate` directory. Files with the `exact` suffix contain
two-sided conditional sign-enumeration Wilcoxon tests. Zero differences are
removed and absolute ties receive average ranks. Enumeration is limited to 20
nonzero pairs; larger samples raise an error rather than changing methods.

The tests use 16 paired VLA-by-generation-strategy blocks. Their interpretation
requires valid sign changes under the null; an exact calculation does not
establish independence or symmetry. The normal approximation is available with
`--wilcoxon-method approximate`. In this mode, Holm still covers all 561 pairs;
the original 33-comparison analysis remains in the unsuffixed historical files.

Generate the corresponding tables and their separate manifest with:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src_new python -B experiments/RQ1/generate_tables.py \
  --statistical-review-method exact
```

The exact reference table shows the 33 Harmonic Mean comparisons, but its Holm
values come from the complete family of 561 unordered pairs. The full family is
retained in `pairwise_comparisons_exact.csv`. The historical `manifest.json` and
historical table files are not modified by this mode.

The reference table reports ROC-AUC, mean gaps, p-values, paired rank-biserial
effects and significance decisions. It does not include confidence intervals.
The original hierarchical-bootstrap intervals remain available in
`aggregate/statistical_comparison.csv`; they are pointwise intervals and are
neither Holm-adjusted nor adjusted for selection of a reference technique.
The `practically_equivalent` field is a descriptive rule combining an absolute
mean gap below 0.01 with non-significance; it is not a formal equivalence test.

`ranking_sensitivity.csv` and `ranking_sensitivity_summary.csv` contain complete
leave-one-VLA-out, leave-one-strategy-out and within-group descriptive rankings.
These overlapping subsets are not treated as independent inferential samples.
