# Trajectory-level baseline comparison: paper RQ2

This directory implements RQ2 of the paper. The voxel-size study is stored
separately in `experiments/RQ2` and corresponds to RQ3 of the paper.

The complete four-model baseline results are in `results/all_models`.
`results/baseline_comparison` contains the earlier execution and is not the
source for the complete four-model comparison.

## Statistical analysis from aggregate results

Run these commands from the repository root with the dependencies declared in
`src_new/pyproject.toml` to calculate tests from the stored aggregate CSVs:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src_new python -B experiments/RQ2_baselines/run_experiment.py --output-root experiments/RQ2_baselines/results/all_models --statistical-review-only --wilcoxon-method exact
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src_new python -B experiments/RQ2_baselines/generate_tables.py --aggregate-dir experiments/RQ2_baselines/results/all_models/aggregate --output-dir experiments/RQ2_baselines/results/all_models/tables --statistical-review-method exact
```

Use `approximate` in both commands to select the normal approximation.
Both modes write method-labelled review files and separate manifests; they do not
rebuild folds, load trajectories, refit models, or regenerate bootstrap intervals.
Historical files without method suffixes remain the original analysis.
The exact method reuses RQ1's conditional two-sided sign enumeration (zeros removed,
average ranks for absolute ties, maximum 20 nonzero pairs). Its inference requires
valid independent sign changes under the symmetric paired-difference null.
The original experimental runners retain their approximate default.

Holm correction covers the eight baseline comparisons against Harmonic Mean.
The observations are 16 paired VLA-by-generation-strategy blocks, with folds
averaged within each budget and the two budgets equally weighted within blocks.
The retained hierarchical-bootstrap intervals are pointwise, not simultaneous,
Holm-adjusted or adjusted for selection. `practically_equivalent` denotes the
descriptive rule of an absolute mean ROC-AUC gap below 0.01 and non-significance
after Holm; it does not establish formal equivalence.

The analysis compares the RQ1 Harmonic Mean result with uncertainty and quality
metrics stored in each raw episode's `allMetrics/0` directory.

## Files

- `RQ2.md`: scientific protocol and interpretation boundary.
- `catalogue.py`: the 13 metric files, display names, directions, and observability.
- `run_experiment.py`: extraction, provenance audit, evaluation, aggregation, and
  paired statistics.
- `analysis.py`: paired statistical helpers shared with the RQ1 conventions.
- `generate_tables.py`: CSV and LaTeX table generation from completed aggregates.
- `EXECUTION.md`: experiment execution commands.

Raw VLA data are inputs and are never modified. Empty or missing metric files are
reported as unavailable; the runner does not silently discard affected episodes.
