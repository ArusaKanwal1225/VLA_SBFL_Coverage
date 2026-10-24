# RQ3 execution guide

RQ3 evaluates how voxel coverage affects the 31 SBFL techniques while holding
the number and class balance of training trajectories fixed. The primary grid
uses voxel size `0.02`.

## Smoke test

From the repository root:

```bash
PYTHONPATH="$PWD:$PWD/src_new" python experiments/RQ3/run_experiment.py --smoke
```

This runs one `EO1/DH/200` fold, one training size, and the three coverage
tiers. It writes only to `experiments/RQ3/results/smoke`.

## Full primary experiment

```bash
PYTHONPATH="$PWD:$PWD/src_new" python experiments/RQ3/run_experiment.py
```

The default run is resumable and writes to
`experiments/RQ3/results/phase_a_coverage`. Models and prediction tables are not
saved by default because the complete design contains 4,800 fitted subset
cells. Selected episode IDs, candidate-bank coverage, cell metrics, and hashes
are retained, so cells are reproducible.

Use `--save-models` or `--save-predictions` only when those large artifacts are
needed.

## Voxel-size robustness

Run the same design at all three RQ2 sizes in a separate output root:

```bash
PYTHONPATH="$PWD:$PWD/src_new" python experiments/RQ3/run_experiment.py \
  --voxel-size 0.01 --voxel-size 0.02 --voxel-size 0.05 \
  --output-root experiments/RQ3/results/size_robustness
```

## Analysis only

```bash
PYTHONPATH="$PWD:$PWD/src_new" python experiments/RQ3/run_experiment.py \
  --analysis-only
```

Important aggregate outputs are:

- `aggregate/cell_metrics.csv.gz`: every selected subset/formula result.
- `aggregate/coverage_tier_metrics.csv`: results by training size and tier.
- `aggregate/rankings_by_coverage.csv`: technique rankings for each tier.
- `aggregate/matched_coverage_effects.csv`: paired high-minus-low effects.
- `aggregate/technique_coverage_interactions.csv`: interactions against
  Harmonic Mean.
- `aggregate/RESULTS.md`: concise generated summary.
- `aggregate/summary.json`: machine-readable completion and leaders.
