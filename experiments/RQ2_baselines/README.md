# RQ2: trajectory-level baseline comparison

This directory implements the new RQ2 inserted after the SBFL-technique study.
It compares the frozen RQ1 Harmonic Mean result with the uncertainty and quality
metrics stored in each raw episode's `allMetrics/0` directory.

The existing `experiments/RQ2` directory has intentionally not been renamed or
overwritten because it currently contains the voxel-size study and uncommitted
results. Once the paper and repository are renumbered together, that study should
become RQ3 and this directory can become `experiments/RQ2`.

## Files

- `RQ2.md`: frozen scientific protocol and interpretation boundary.
- `catalogue.py`: the 13 metric files, display names, directions, and observability.
- `run_experiment.py`: extraction, provenance audit, evaluation, aggregation, and
  paired statistics.
- `analysis.py`: paired statistical helpers shared with the RQ1 conventions.
- `generate_tables.py`: CSV and LaTeX table generation from completed aggregates.
- `EXECUTION.md`: commands for EO1 and the future four-VLA run.

Raw VLA data are inputs and are never modified. Empty or missing metric files are
reported as unavailable; the runner does not silently discard affected episodes.
