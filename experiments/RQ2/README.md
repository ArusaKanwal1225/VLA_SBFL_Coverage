# RQ2 execution guide

RQ2 compares voxel sizes `0.01`, `0.02`, and `0.05` using exactly the RQ1 folds
and all 31 spectrum-based techniques. The completed RQ1 `0.02` cell is validated
and reused. The runner computes the 640 new folds for `0.01` and `0.05`, then
performs the matched cross-size analysis.

## Full execution

From the repository root, with the project dependencies installed:

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

The smoke test runs one held-out fold at each new size and writes to a separate
directory, so it cannot be mistaken for the confirmatory output:

```bash
PYTHONPATH="$PWD:$PWD/src_new" python experiments/RQ2/run_experiment.py \
  --smoke --bootstrap-iterations 200
```

## Outputs

The full default output root is `experiments/RQ2/results/phase_a_granularity`.
Important aggregate files are:

- `aggregate/voxel_size_comparison.csv`: confirmatory Harmonic Mean size test.
- `aggregate/rankings_by_size.csv`: all 31 techniques at every size.
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
