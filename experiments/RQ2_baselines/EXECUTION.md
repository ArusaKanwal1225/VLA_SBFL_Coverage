# RQ2 execution guide

Run commands from the repository root with both the repository and `src_new` on
`PYTHONPATH`.

## EO1

The checked-out raw dataset is discovered automatically:

```bash
PYTHONPATH="$PWD:$PWD/src_new" python experiments/RQ2_baselines/run_experiment.py \
  --vla EO1
```

The run is resumable. It evaluates all eight EO1 configurations and 80 matched
runs, audits metric completeness, combines the results with RQ1 Harmonic Mean,
and performs the paired analysis.

To rebuild only the aggregate statistics after all folds have completed:

```bash
PYTHONPATH="$PWD:$PWD/src_new" python experiments/RQ2_baselines/run_experiment.py \
  --vla EO1 --analysis-only
```

Use a separate output for a one-run smoke test:

```bash
PYTHONPATH="$PWD:$PWD/src_new" python experiments/RQ2_baselines/run_experiment.py \
  --vla EO1 --max-configurations 1 --max-runs 1 --extract-only \
  --output-root experiments/RQ2_baselines/results/smoke
```

## All VLA models

Place the raw model directories beneath one `Data` directory, preserving the
configuration/run/result hierarchy, and run:

```bash
PYTHONPATH="$PWD:$PWD/src_new" python experiments/RQ2_baselines/run_experiment.py \
  --raw-root /path/to/Data \
  --vla EO1 --vla Groot --vla Pi0 --vla SpatialVLA
```

Recognized raw directory aliases include `GR00T` for the compact-input name
`Groot`. The actual model directory between a scene and `allMetrics/0` is
discovered rather than hard-coded.

## Tables

After the complete experiment:

```bash
PYTHONPATH="$PWD:$PWD/src_new" python experiments/RQ2_baselines/generate_tables.py
```

Important aggregate outputs are:

- `aggregate/metric_availability.csv`
- `aggregate/performance_by_vla.csv`
- `aggregate/overall_performance.csv`
- `aggregate/statistical_comparison.csv`
- `aggregate/statistical_tests.json`
- `aggregate/summary.json`
