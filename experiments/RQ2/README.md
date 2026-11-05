# Voxel-size sensitivity: paper RQ3

This directory is `experiments/RQ2` in the source tree and implements RQ3 of the paper.

## Statistical analysis from aggregate results

Run these commands from the repository root with the dependencies declared in
`src_new/pyproject.toml` to calculate tests from the stored aggregate CSVs:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src_new python -B experiments/RQ2/run_experiment.py --output-root experiments/RQ2/results/phase_a_granularity --statistical-review-only --wilcoxon-method exact
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src_new python -B experiments/RQ2/generate_tables.py --aggregate-dir experiments/RQ2/results/phase_a_granularity/aggregate --output-dir experiments/RQ2/results/phase_a_granularity/tables --statistical-review-method exact
```

Use `approximate` in both commands to select the normal approximation.
Both modes write method-labelled review files and separate manifests; they do not
rebuild folds, load trajectories, refit models, or regenerate bootstrap intervals.
Historical files without method suffixes remain the original analysis.
The exact method reuses RQ1's conditional two-sided sign enumeration (zeros removed,
average ranks for absolute ties, maximum 20 nonzero pairs). Its inference requires
valid independent sign changes under the symmetric paired-difference null.
The original experimental runners retain their approximate default.

Holm correction is applied separately to 68 technique-size effects and 66
technique-size interactions. The primary voxel size is 0.02. The Friedman
analysis of Harmonic Mean across sizes is recorded separately in
`aggregate/statistical_tests.json`.

The paired observations are the 16 VLA-by-generation-strategy blocks. Statistical
inference requires valid sign changes under the null; exact enumeration does not
establish block independence. Any retained bootstrap intervals are pointwise,
not simultaneous, Holm-adjusted or adjusted for selection. Meaningfulness labels
combine observed mean differences with corrected test decisions; they are not
formal equivalence tests.

The voxel-size experiment compares `0.01`, `0.02`, and `0.05` using the RQ1 folds
and all 34 spectrum-based techniques. The completed RQ1 `0.02` cell is validated
and reused. The runner computes the 640 new folds for `0.01` and `0.05`, then
performs the matched cross-size analysis.

## Full execution

From the repository root, with the project dependencies installed:

The commands in this section execute experiment cells or rebuild the original
aggregate files. They require the input data and matching experiment signatures.
Use a separate output directory when running a new experiment; use the
aggregate-only commands above to reproduce the exact statistical results.

```bash
PYTHONPATH="$PWD:$PWD/src_new" python experiments/RQ2/run_experiment.py
```

The run is resumable by default. Repeating the command skips completed folds
after checking their experiment signatures.

To recompute the `0.02` cell instead of reusing RQ1:

```bash
PYTHONPATH="$PWD:$PWD/src_new" python experiments/RQ2/run_experiment.py \
  --rerun-primary
```

To perform only the final cross-size analysis after all cells exist:

```bash
PYTHONPATH="$PWD:$PWD/src_new" python experiments/RQ2/run_experiment.py \
  --analysis-only
```

## Smoke test

The smoke test runs one held-out fold at each new size and writes outside the
full-experiment output directory:

```bash
PYTHONPATH="$PWD:$PWD/src_new" python experiments/RQ2/run_experiment.py \
  --smoke --bootstrap-iterations 200
```

## Outputs

The full default output root is `experiments/RQ2/results/phase_a_granularity`.
The list below describes the default experiment outputs. The aggregate-only
commands above produce separate files with `_exact.csv` or `_approximate.csv`
suffixes and method-specific JSON metadata.

- `aggregate/voxel_size_comparison.csv`: Harmonic Mean size comparison.
- `aggregate/rankings_by_size.csv`: all 34 techniques at every size.
- `aggregate/technique_size_effects.csv`: each formula's change from `0.02`.
- `aggregate/technique_size_interactions.csv`: difference-in-differences against
  Harmonic Mean with Holm correction.
- `aggregate/ranking_stability.csv`: cross-size Spearman rank correlations.
- `aggregate/size_diagnostics.csv`: occupied voxels, coverage, ties, runtime, and
  artifact size.
- `aggregate/RESULTS.md`: concise generated result summary.
- `aggregate/summary.json`: machine-readable experiment conclusion.

The `0.01` artifacts may require substantially more disk space than the current
`0.02` run because they can contain more occupied training voxels.

The exact statistical tables use the LaTeX `booktabs` and `longtable` packages.
